//! CPU streaming benchmark of a Small step graph. PIPELINE-ONLY unless run on a real checkpoint.
//!
//!   cargo run --release -p tingmix-runtime-onnx --example bench_cpu -- <model.onnx> <out.json> [label] [seconds]
//!
//! Runs every (threads, chunk) pair in its own child process (so peak RSS is per configuration),
//! feeding `seconds` of fixture speech in chunk-sized pushes as fast as possible, and writes one
//! JSON document with the command, hardware, model sha256 and per-configuration results.
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tingmix_runtime_onnx::{StepModel, Streamer};

const THREADS: [usize; 4] = [1, 2, 4, 8];
const CHUNKS_MS: [usize; 3] = [160, 320, 640];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn audio(seconds: usize) -> Vec<f32> {
    let mut all = Vec::new();
    for f in ["mix_29s", "zh_16s", "en_08s", "zh_08s"].iter().cycle() {
        let mut r =
            hound::WavReader::open(root().join(format!("tests/fixtures/audio/{f}.wav"))).unwrap();
        all.extend(r.samples::<i16>().map(|s| f32::from(s.unwrap()) / 32768.0));
        if all.len() >= seconds * 16_000 {
            break;
        }
    }
    all.truncate(seconds * 16_000);
    all
}

#[cfg(unix)]
fn peak_rss_mb() -> f64 {
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
    // macOS reports bytes, Linux kilobytes
    let bytes = if cfg!(target_os = "macos") {
        u.ru_maxrss as f64
    } else {
        u.ru_maxrss as f64 * 1024.0
    };
    bytes / 1e6
}

#[cfg(windows)]
fn peak_rss_mb() -> f64 {
    #[repr(C)]
    struct Counters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        rest: [usize; 7],
    }
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut Counters, cb: u32) -> i32;
    }
    let mut c = Counters {
        cb: std::mem::size_of::<Counters>() as u32,
        page_fault_count: 0,
        peak_working_set_size: 0,
        rest: [0; 7],
    };
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.peak_working_set_size as f64 / 1e6 // peak working set
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn single(model: &Path, threads: usize, chunk_ms: usize, seconds: usize) {
    let wav = audio(seconds);
    let m = Arc::new(Mutex::new(StepModel::load(model, threads).unwrap()));
    let mut s = Streamer::new(m);
    let chunk = chunk_ms * 16;
    let (mut times, mut first_out_ms) = (Vec::new(), None);
    let start = Instant::now();
    for (i, c) in wav.chunks(chunk).enumerate() {
        let t0 = Instant::now();
        let out = s.push(c).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if !out.is_empty() && first_out_ms.is_none() {
            first_out_ms = Some(dt);
        }
        if i > 0 {
            times.push(dt); // the first push pays session warm-up
        }
    }
    let total = start.elapsed().as_secs_f64();
    times.sort_by(f64::total_cmp);
    println!(
        "{{\"threads\":{threads},\"chunk_ms\":{chunk_ms},\"audio_s\":{:.2},\"wall_s\":{total:.3},\"rtf\":{:.4},\
\"push_ms_p50\":{:.3},\"push_ms_p95\":{:.3},\"push_ms_p99\":{:.3},\"push_ms_max\":{:.3},\"first_output_push_ms\":{:.3},\
\"pushes_slower_than_chunk\":{},\"peak_rss_mb\":{:.1}}}",
        wav.len() as f64 / 16e3,
        total / (wav.len() as f64 / 16e3),
        pct(&times, 0.5),
        pct(&times, 0.95),
        pct(&times, 0.99),
        times.last().unwrap(),
        first_out_ms.unwrap_or(f64::NAN),
        times.iter().filter(|&&t| t > chunk_ms as f64).count(),
        peak_rss_mb()
    );
}

fn sh(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default()
}

fn ps(script: &str) -> String {
    sh(
        "powershell",
        &["-NoProfile", "-NonInteractive", "-Command", script],
    )
}

fn sha256(path: &str) -> String {
    if cfg!(windows) {
        ps(&format!(
            "(Get-FileHash -Algorithm SHA256 -LiteralPath '{path}').Hash.ToLower()"
        ))
    } else {
        sh("shasum", &["-a", "256", path])
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_owned()
    }
}

/// (cpu, logical cpus, memory bytes, os)
fn hardware() -> (String, String, String, String) {
    if cfg!(windows) {
        (
            ps("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"),
            ps("[Environment]::ProcessorCount"),
            ps("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"),
            ps("(Get-CimInstance Win32_OperatingSystem | ForEach-Object { $_.Caption + ' ' + $_.Version })"),
        )
    } else {
        (
            sh("sysctl", &["-n", "machdep.cpu.brand_string"]),
            sh("sysctl", &["-n", "hw.ncpu"]),
            sh("sysctl", &["-n", "hw.memsize"]),
            sh("sw_vers", &["-productVersion"]),
        )
    }
}

/// Which ONNX Runtime the numbers came from.
fn ort_label() -> String {
    match std::env::var("TINGMIX_ORT_DYLIB").or_else(|_| std::env::var("ORT_DYLIB_PATH")) {
        Ok(p) if !p.is_empty() => format!(
            "load-dynamic: {p} ({})",
            if cfg!(windows) {
                ps(&format!(
                    "(Get-Item -LiteralPath '{p}').VersionInfo.ProductVersion"
                ))
            } else {
                String::new()
            }
        ),
        _ => "1.31.0 (ort 2.0.0-rc.14 prebuilt, CPU EP)".into(),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(String::as_str) == Some("--single") {
        single(
            Path::new(&a[2]),
            a[3].parse().unwrap(),
            a[4].parse().unwrap(),
            a[5].parse().unwrap(),
        );
        return;
    }
    let (model, out) = (&a[1], &a[2]);
    let label = a
        .get(3)
        .cloned()
        .unwrap_or_else(|| "pipeline-only (random-init weights)".into());
    let seconds: usize = a.get(4).map_or(60, |s| s.parse().unwrap());
    let exe = std::env::current_exe().unwrap();
    let mut rows = Vec::new();
    for t in THREADS {
        for c in CHUNKS_MS {
            let o = Command::new(&exe)
                .args([
                    "--single",
                    model,
                    &t.to_string(),
                    &c.to_string(),
                    &seconds.to_string(),
                ])
                .output()
                .unwrap();
            assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
            let line = String::from_utf8_lossy(&o.stdout).trim().to_owned();
            println!("{line}");
            rows.push(line);
        }
    }
    let (cpu, ncpu, mem, os) = hardware();
    let doc = format!(
        "{{\n \"label\":{label:?},\n \"command\":{:?},\n \"model\":{model:?},\n \"model_sha256\":{:?},\n \"onnxruntime\":{:?},\n \"hardware\":{{\"cpu\":{cpu:?},\"logical_cpus\":{ncpu:?},\"memory_bytes\":{mem:?},\"os\":{os:?}}},\n \"results\":[\n  {}\n ]\n}}\n",
        a.join(" "),
        sha256(model),
        ort_label(),
        rows.join(",\n  ")
    );
    std::fs::write(out, doc).unwrap();
}
