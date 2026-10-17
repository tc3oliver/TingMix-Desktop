//! `ProbeTarget` for the MLX backend. Decoding runs in a long-lived child (`tingmix-mlx-worker`) so a Metal/MLX crash kills the
//! child, not the app: the parent sees a closed pipe or a timeout and reports an error, and the next call respawns the child.
//! The child loads the model once (first decode), so `run_probe`'s timed runs measure warm decode plus pipe transfer.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;

use tingmix_models::probe::ProbeTarget;
use tingmix_models::InstalledModel;

use crate::Dtype;

struct Worker {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

pub struct MlxProbeTarget {
    exe: PathBuf,
    dtype: Dtype,
    use_lm: bool,
    timeout: Duration,
    worker: Option<Worker>,
    info: Option<(String, String)>,
}

impl MlxProbeTarget {
    /// `worker_exe`: path to the `tingmix-mlx-worker` binary. `use_lm`: probe with beam+LM (the production decode) instead of greedy.
    pub fn new(worker_exe: PathBuf, dtype: Dtype, use_lm: bool) -> Self {
        Self {
            exe: worker_exe,
            dtype,
            use_lm,
            timeout: Duration::from_secs(120),
            worker: None,
            info: None,
        }
    }

    /// `tingmix-mlx-worker` next to the running executable.
    pub fn sibling_worker() -> std::io::Result<PathBuf> {
        Ok(std::env::current_exe()?.with_file_name("tingmix-mlx-worker"))
    }

    fn spawn(&mut self, model: &InstalledModel) -> Result<(), String> {
        let first = |c: &str| {
            model
                .file(c)
                .next()
                .ok_or_else(|| format!("manifest has no {c} file"))
        };
        let artifact = first("model")?;
        let units = model
            .file("tokenizer")
            .find(|p| p.file_name().is_some_and(|n| n == "units.json"))
            .ok_or("manifest has no units.json")?;
        let mut cmd = Command::new(&self.exe);
        cmd.arg(artifact).arg(units).arg(match self.dtype {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
        });
        if self.use_lm {
            cmd.arg(first("lm")?);
        }
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", self.exe.display()))?;
        let (stdin, stdout) = (child.stdin.take().unwrap(), child.stdout.take().unwrap());
        let (tx, lines) = channel();
        std::thread::spawn(move || {
            for l in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx.send(l).is_err() {
                    break;
                }
            }
        });
        let mut w = Worker {
            child,
            stdin,
            lines,
        };
        match w.lines.recv_timeout(self.timeout) {
            Ok(l) => match l.split('\t').collect::<Vec<_>>()[..] {
                ["READY", v, d] => self.info = Some((v.into(), d.into())),
                _ => {
                    let _ = w.child.kill();
                    return Err(format!("worker failed to start: {l}"));
                }
            },
            Err(_) => {
                let _ = w.child.kill();
                return Err("worker did not start (crashed or timed out)".into());
            }
        }
        self.worker = Some(w);
        Ok(())
    }
}

impl Drop for MlxProbeTarget {
    fn drop(&mut self) {
        if let Some(mut w) = self.worker.take() {
            drop(w.stdin);
            let _ = w.child.wait();
        }
    }
}

impl ProbeTarget for MlxProbeTarget {
    fn backend(&self) -> &str {
        "mlx"
    }

    /// Known after the first successful decode (the version comes from the loaded libmlx inside the child).
    fn backend_version(&self) -> Option<String> {
        self.info.as_ref().map(|i| i.0.clone())
    }

    fn device(&self) -> String {
        self.info.as_ref().map(|i| i.1.clone()).unwrap_or_default()
    }

    fn decode(&mut self, model: &InstalledModel, wav: &[f32]) -> Result<String, String> {
        if self.worker.is_none() {
            self.spawn(model)?;
        }
        let w = self.worker.as_mut().unwrap();
        let mut req = Vec::with_capacity(8 + wav.len() * 4);
        req.extend_from_slice(&(wav.len() as u64).to_le_bytes());
        wav.iter()
            .for_each(|s| req.extend_from_slice(&s.to_le_bytes()));
        let sent = w.stdin.write_all(&req).and_then(|_| w.stdin.flush());
        let reply = match sent {
            Ok(()) => w
                .lines
                .recv_timeout(self.timeout)
                .map_err(|_| "worker timed out or crashed".to_string()),
            Err(e) => Err(format!("worker pipe: {e}")),
        };
        match reply {
            Ok(l) => match l.split_once('\t') {
                Some(("OK", t)) => Ok(t.into()),
                Some(("ERR", m)) => Err(m.into()),
                _ => Err(format!("bad worker reply: {l}")),
            },
            Err(e) => {
                if let Some(mut w) = self.worker.take() {
                    let _ = w.child.kill(); // respawned by the next call
                    let _ = w.child.wait();
                }
                Err(e)
            }
        }
    }
}
