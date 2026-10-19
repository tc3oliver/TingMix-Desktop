//! First-run setup: when no model is installed, find the one for this computer in the signed catalog and offer it. The
//! download, signature check, activation and compatibility test then run through the same path as any other model
//! (`Models::install` -> `backends::onboard_core`), so the screen only reports what the host is already doing.

use std::sync::Arc;

use serde::Serialize;
use serde_json::json;
use tingmix_models::{catalog, select, VerifiedManifest};

use crate::backends::{self, Runtimes};
use crate::coded::coded;
use crate::models::{Models, PendingInstall};

/// The signed catalog, fixed at build time: a JSON array of manifests signed with the release key, each with file URLs pinned
/// to a commit, sizes and hashes. Fetching it from `main` is safe: the signatures and serials protect it. Entries are
/// filtered by OS and CPU after the fetch, so a platform without a package (Windows x86_64 until task 28) gets "not
/// available for this computer yet".
pub const DEFAULT_CATALOG_URL: &str =
    "https://huggingface.co/tc3oliver/TingMix-Desktop/resolve/main/catalog.json";

/// `TINGMIX_CATALOG_URL` at compile time overrides the default (builds that test another server).
pub fn catalog_url() -> &'static str {
    option_env!("TINGMIX_CATALOG_URL").unwrap_or(DEFAULT_CATALOG_URL)
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum SetupView {
    /// A real engine is chosen and loaded or loading: nothing to set up.
    Ready,
    /// No model is installed; this one fits the computer. The licences must be accepted before `models_install`.
    Download { pending: PendingInstall },
    /// A model is installed but cannot be used (test failed, engine missing in this build, nothing selected).
    Problem { reason: String },
    /// Nothing in the catalog fits: `platform` (no entry for this OS/CPU or not downloadable) or `build` (this build lacks the engine).
    Unavailable { why: String, reason: String },
    /// The catalog could not be fetched or verified.
    Offline { reason: String },
}

pub struct Catalog<'a> {
    pub url: &'a str,
    pub allow_loopback_http: bool,
}

fn no_platform() -> SetupView {
    SetupView::Unavailable {
        why: "platform".into(),
        reason: coded(
            "setup.no_platform",
            json!({}),
            "The speech model is not available for this computer yet.",
        ),
    }
}

pub fn check(
    models: &Arc<Models>,
    rt: &Runtimes,
    catalog_cfg: &Catalog,
    token: String,
) -> SetupView {
    let info = backends::info(models, rt);
    if !info.mock {
        return SetupView::Ready;
    }
    // an install whose manifest this build cannot trust or parse (for example signed by a key it no longer trusts) does not count: the catalog
    // offers a verified replacement for it, and the screen would otherwise sit in Problem for good
    let installed: Vec<String> = models
        .reg()
        .ok()
        .and_then(|r| {
            r.list().ok().map(|ids| {
                ids.into_iter()
                    .filter(|id| r.open_model(id).err().is_none_or(|e| !e.replaceable()))
                    .collect()
            })
        })
        .unwrap_or_default();
    if !installed.is_empty() {
        return SetupView::Problem {
            reason: info.reason,
        };
    }
    let downloadable = match offers(models, catalog_cfg) {
        Ok(d) => d,
        Err(reason) => return SetupView::Offline { reason },
    };
    if downloadable.is_empty() {
        return no_platform();
    }
    // MLX is the faster engine on Apple Silicon; do not let catalog order decide
    let Some(m) = downloadable
        .into_iter()
        .filter(|m| backends::engine_ready(&m.manifest().backend, rt))
        .min_by_key(|m| m.manifest().backend != backends::MLX)
    else {
        return SetupView::Unavailable {
            why: "build".into(),
            reason: coded(
                "setup.no_engine",
                json!({}),
                "This build of TingMix does not include the engine the speech model needs. Install the full build.",
            ),
        };
    };
    match models.begin_verified(m, token) {
        Ok(pending) => SetupView::Download { pending },
        Err(e) => SetupView::Offline { reason: e },
    }
}

/// Catalog entries this computer can download and run once its engine is present: built for this OS and CPU, every file
/// downloadable, requirements met, and not older than what the registry has seen (the same rule as `Registry::prepare`, so an
/// offered entry is never refused as a replay after the download). `Err` is a coded "cannot reach the server" reason.
fn offers(models: &Arc<Models>, catalog_cfg: &Catalog) -> Result<Vec<VerifiedManifest>, String> {
    let reg = models.reg()?;
    let cat = catalog::fetch(
        catalog_cfg.url,
        reg.verifier(),
        catalog_cfg.allow_loopback_http,
    )
    .map_err(|e| {
        let detail = e.to_string();
        coded(
            "setup.offline",
            json!({ "detail": detail }),
            format!("TingMix cannot reach the model server: {detail}"),
        )
    })?;
    let caps = models.caps();
    let replayed = |m: &tingmix_models::Manifest| {
        reg.high_water(&m.lineage())
            .ok()
            .is_some_and(|(serial, id)| {
                m.serial < serial || (m.serial == serial && id.is_some_and(|id| id != m.id))
            })
    };
    Ok(cat
        .for_platform(&caps.os, &caps.arch)
        .into_iter()
        .filter(|m| {
            let mf = m.manifest();
            mf.files.iter().all(|f| !f.url.is_empty())
                && select::requirements_met(mf, caps, env!("CARGO_PKG_VERSION")).is_ok()
                && !replayed(mf)
        })
        .cloned()
        .collect())
}

/// One model the user can pick from the catalog.
#[derive(Debug, Clone, Serialize)]
pub struct Offer {
    pub id: String,
    pub family: String,
    pub backend: String,
    pub precision: String,
    pub download_bytes: u64,
    pub compat_status: String,
    pub installed: bool,
    /// This build has the engine (and its test helper) the model needs.
    pub engine_ready: bool,
}

/// Every catalog model for this computer, for the model picker.
pub fn list(
    models: &Arc<Models>,
    rt: &Runtimes,
    catalog_cfg: &Catalog,
) -> Result<Vec<Offer>, String> {
    let installed = models.reg()?.list().unwrap_or_default();
    Ok(offers(models, catalog_cfg)?
        .iter()
        .map(|m| {
            let mf = m.manifest();
            Offer {
                id: mf.id.clone(),
                family: mf.family.clone(),
                backend: mf.backend.clone(),
                precision: mf.precision.clone(),
                download_bytes: mf.files.iter().map(|f| f.size).sum(),
                compat_status: mf.compat.status.clone(),
                installed: installed.contains(&mf.id),
                engine_ready: backends::engine_ready(&mf.backend, rt),
            }
        })
        .collect())
}

/// Start installing catalog model `id`: the catalog is fetched and verified again, so the id is all the webview supplies.
pub fn begin(
    models: &Arc<Models>,
    catalog_cfg: &Catalog,
    id: &str,
    token: String,
) -> Result<PendingInstall, String> {
    let m = offers(models, catalog_cfg)?
        .into_iter()
        .find(|m| m.manifest().id == id)
        .ok_or_else(|| format!("{id} is not offered for this computer"))?;
    models.begin_verified(m, token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Notify;
    use ed25519_dalek::SigningKey;
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};
    use tingmix_models::manifest::*;
    use tingmix_models::Verifier;

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn verifier() -> Verifier {
        Verifier::new(&[("test", key(1).verifying_key().to_bytes())]).unwrap()
    }

    fn blob(n: usize, seed: u8) -> Vec<u8> {
        let mut x = seed as u32 + 1;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                (x >> 24) as u8
            })
            .collect()
    }

    /// A manifest for this computer whose files are served at `base`.
    fn manifest(
        id: &str,
        base: &str,
        os: &str,
        backend: &str,
    ) -> (Manifest, HashMap<String, Vec<u8>>) {
        let parts: Vec<(&str, &str, Vec<u8>)> = vec![
            ("model.safetensors", "reader", blob(200_000, 3)),
            ("units.json", "tokenizer", b"[\"<blank>\",\"a\"]".to_vec()),
        ];
        let mut served = HashMap::new();
        let files = parts
            .into_iter()
            .map(|(p, c, b)| {
                served.insert(format!("/{id}/{p}"), b.clone());
                FileEntry {
                    path: p.into(),
                    component: c.into(),
                    url: format!("{base}/{id}/{p}"),
                    size: b.len() as u64,
                    sha256: sha256_hex(&b),
                }
            })
            .collect();
        let m = Manifest {
            schema_version: 2,
            serial: 1,
            id: id.into(),
            family: "tingmix-large".into(),
            revision: "0".repeat(40),
            backend: backend.into(),
            os: os.into(),
            arch: std::env::consts::ARCH.into(),
            precision: "fp32".into(),
            requirements: Requirements {
                min_ram_bytes: 0,
                cpu_features: vec![],
                apple_chip_min_generation: None,
            },
            runtime: RuntimeReq {
                name: backend.into(),
                min_version: "0".into(),
            },
            base_model: None,
            adapter: None,
            tokenizer: None,
            lm: None,
            files,
            licenses: vec![LicenseTerms {
                id: "terms".into(),
                name: "Test terms".into(),
                spdx: None,
                url: "https://example.invalid/terms".into(),
                applies_to: vec!["reader".into(), "tokenizer".into()],
                acceptance_required: true,
                attribution: vec![],
                notes: String::new(),
            }],
            compat: Compat {
                status: "EXPERIMENTAL".into(),
                rtf_x1000: None,
                notes: String::new(),
            },
            provenance: None,
            created: "2026-10-18".into(),
            min_app_version: "0.1.0".into(),
        };
        (m, served)
    }

    /// Serves `docs` (path -> body) on an already bound loopback listener.
    fn serve_on(l: TcpListener, docs: HashMap<String, Vec<u8>>) {
        std::thread::spawn(move || {
            for conn in l.incoming() {
                let Ok(mut c) = conn else { continue };
                let mut r = BufReader::new(c.try_clone().unwrap());
                let mut line = String::new();
                if r.read_line(&mut line).is_err() {
                    continue;
                }
                let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
                loop {
                    let mut h = String::new();
                    if r.read_line(&mut h).unwrap_or(0) == 0 || h.trim().is_empty() {
                        break;
                    }
                }
                match docs.get(&path) {
                    Some(b) => {
                        let _ = c.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", b.len()).as_bytes());
                        let _ = c.write_all(b);
                    }
                    None => {
                        let _ = c.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                    }
                }
            }
        });
    }

    /// Binds a loopback port first so manifests can name their own file URLs; returns the listener and its base URL.
    fn bind() -> (TcpListener, String) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://127.0.0.1:{}", l.local_addr().unwrap().port());
        (l, base)
    }

    fn serve(docs: HashMap<String, Vec<u8>>) -> String {
        let (l, base) = bind();
        serve_on(l, docs);
        base
    }

    fn catalog_doc(entries: &[&Manifest]) -> Vec<u8> {
        serde_json::to_vec(
            &entries
                .iter()
                .map(|m| sign(m, "test", &key(1)))
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    fn models(tag: &str) -> (Arc<Models>, PathBuf) {
        let dir = std::env::temp_dir().join(format!("tingmix-setup-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        (
            Arc::new(Models::open_with(dir.join("models"), verifier(), true)),
            dir,
        )
    }

    fn onnx_runtime() -> Runtimes {
        Runtimes {
            ort_library: Some(std::env::current_exe().unwrap()),
            // the engine counts as available only with its probe worker; any existing file stands in
            onnx_worker: Some(std::env::current_exe().unwrap()),
        }
    }

    #[test]
    fn first_run_offers_the_download_then_installs_activates_and_tests_it() {
        let (models, dir) = models("flow");
        let (listener, base) = bind();
        let (m, mut docs) = manifest("model-a", &base, std::env::consts::OS, "onnx-cpu");
        docs.insert("/catalog.json".into(), catalog_doc(&[&m]));
        serve_on(listener, docs);
        let url = format!("{base}/catalog.json");
        let rt = onnx_runtime();
        let cfg = Catalog {
            url: &url,
            allow_loopback_http: true,
        };

        let view = check(&models, &rt, &cfg, "t1".into());
        let SetupView::Download { pending } = view else {
            panic!("expected a download offer, got {view:?}")
        };
        assert_eq!(pending.id, "model-a");
        // checking again reuses the pending entry instead of adding one per check
        let SetupView::Download { pending: again } = check(&models, &rt, &cfg, "t1b".into()) else {
            panic!("expected a download offer")
        };
        assert_eq!(again.token, "t1");
        assert!(pending.download_bytes > 200_000);
        assert!(pending
            .licenses
            .iter()
            .any(|l| l.acceptance_required && !l.accepted));

        // the licence must be accepted first; then the download runs in the background
        let notify: Notify = Arc::new(|_| {});
        assert!(models
            .install("t1", &[], notify.clone())
            .unwrap_err()
            .contains("must be accepted"));
        models.install("t1", &["terms".into()], notify).unwrap();
        let t = Instant::now();
        loop {
            let op = models.overview().op.unwrap();
            if op.state != "running" {
                assert_eq!(
                    (op.kind.as_str(), op.state.as_str()),
                    ("download", "done"),
                    "{:?}",
                    op.message
                );
                break;
            }
            assert!(t.elapsed() < Duration::from_secs(30), "download timed out");
            std::thread::sleep(Duration::from_millis(50));
        }
        let ov = models.overview();
        assert_eq!(ov.installed.len(), 1);
        assert_eq!(ov.installed[0].size_bytes, pending.download_bytes);

        // what the host does when the download finishes: activate the first model and test it. The files are not a real
        // model, so the test is expected to fail, with a message the user can read; the real engine run is the e2e test.
        backends::onboard_core(&models, &rt, "model-a", Arc::new(|_| {})).unwrap();
        assert!(models.overview().installed[0].active);
        let t = Instant::now();
        loop {
            let op = models.overview().op.unwrap();
            if op.kind == "probe" && op.state != "running" {
                assert_eq!(op.state, "failed");
                break;
            }
            assert!(t.elapsed() < Duration::from_secs(30), "probe timed out");
            std::thread::sleep(Duration::from_millis(50));
        }
        // with a model installed the screen now describes the problem instead of offering a second download
        assert!(matches!(
            check(&models, &rt, &cfg, "t2".into()),
            SetupView::Problem { .. }
        ));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_picker_lists_every_model_for_this_computer_and_begins_one_by_id() {
        let (models, dir) = models("picker");
        let (listener, base) = bind();
        let (a, mut docs) = manifest("model-a", &base, std::env::consts::OS, "onnx-cpu");
        let (mut b, docs_b) = manifest("model-b", &base, std::env::consts::OS, "mlx");
        b.family = "other".into();
        let (elsewhere, _) = manifest("model-c", &base, "plan9", "onnx-cpu");
        docs.extend(docs_b);
        docs.insert("/catalog.json".into(), catalog_doc(&[&a, &b, &elsewhere]));
        serve_on(listener, docs);
        let url = format!("{base}/catalog.json");
        let cfg = Catalog {
            url: &url,
            allow_loopback_http: true,
        };
        let offers = list(&models, &onnx_runtime(), &cfg).unwrap();
        let ids: Vec<&str> = offers.iter().map(|o| o.id.as_str()).collect();
        assert_eq!(ids.len(), 2, "{ids:?}");
        assert!(ids.contains(&"model-a") && ids.contains(&"model-b"));
        let a = offers.iter().find(|o| o.id == "model-a").unwrap();
        assert!(a.engine_ready && !a.installed && a.download_bytes > 200_000);
        assert_eq!(
            begin(&models, &cfg, "model-b", "p".into()).unwrap().id,
            "model-b"
        );
        assert!(begin(&models, &cfg, "model-c", "q".into()).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_install_signed_by_an_untrusted_key_gets_the_replacement_offered() {
        let (models, dir) = models("untrusted");
        let (listener, base) = bind();
        // installed earlier from a manifest signed with a key this build does not trust, serial 1
        let (old, mut docs) = manifest("model-a", &base, std::env::consts::OS, "onnx-cpu");
        let vdir = dir.join("models/versions/model-a");
        std::fs::create_dir_all(&vdir).unwrap();
        std::fs::write(
            vdir.join("manifest.signed.json"),
            serde_json::to_vec(&sign(&old, "dev", &key(9))).unwrap(),
        )
        .unwrap();
        // the same id re-signed with the trusted key at a higher serial
        let mut newer = old.clone();
        newer.serial = 6;
        docs.insert("/catalog.json".into(), catalog_doc(&[&newer]));
        serve_on(listener, docs);
        let url = format!("{base}/catalog.json");
        let cfg = Catalog {
            url: &url,
            allow_loopback_http: true,
        };
        let SetupView::Download { pending } = check(&models, &onnx_runtime(), &cfg, "u".into())
        else {
            panic!("expected the replacement to be offered")
        };
        assert_eq!(pending.id, "model-a");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn unreachable_server_unknown_platform_and_missing_engine_each_get_their_own_answer() {
        let (models, dir) = models("states");
        let rt = onnx_runtime();

        assert!(
            DEFAULT_CATALOG_URL.starts_with("https://")
                && DEFAULT_CATALOG_URL.ends_with("/catalog.json")
        );

        let cfg = Catalog {
            url: "http://127.0.0.1:1/catalog.json",
            allow_loopback_http: true,
        };
        assert!(matches!(
            check(&models, &rt, &cfg, "a".into()),
            SetupView::Offline { .. }
        ));

        let (other_os, _) = manifest("m-other", "http://127.0.0.1:9", "plan9", "onnx-cpu");
        let base = serve(HashMap::from([(
            "/c.json".to_string(),
            catalog_doc(&[&other_os]),
        )]));
        let url = format!("{base}/c.json");
        let cfg = Catalog {
            url: &url,
            allow_loopback_http: true,
        };
        let SetupView::Unavailable { why, .. } = check(&models, &rt, &cfg, "b".into()) else {
            panic!("platform")
        };
        assert_eq!(why, "platform");

        // an entry that needs a newer app than this one is not offered
        let (mut newer, _) = manifest(
            "m-newer",
            "http://127.0.0.1:9",
            std::env::consts::OS,
            "onnx-cpu",
        );
        newer.min_app_version = "99.0.0".into();
        let base = serve(HashMap::from([(
            "/n.json".to_string(),
            catalog_doc(&[&newer]),
        )]));
        let url = format!("{base}/n.json");
        let cfg = Catalog {
            url: &url,
            allow_loopback_http: true,
        };
        let SetupView::Unavailable { why, .. } = check(&models, &rt, &cfg, "n".into()) else {
            panic!("min_app_version")
        };
        assert_eq!(why, "platform");

        // fits the OS, but this build has no such engine
        let (mlx, _) = manifest("m-mlx", "http://127.0.0.1:9", std::env::consts::OS, "mlx");
        let base = serve(HashMap::from([(
            "/c.json".to_string(),
            catalog_doc(&[&mlx]),
        )]));
        let url = format!("{base}/c.json");
        let cfg = Catalog {
            url: &url,
            allow_loopback_http: true,
        };
        if !backends::engine_ready("mlx", &rt) {
            let SetupView::Unavailable { why, .. } = check(&models, &rt, &cfg, "c".into()) else {
                panic!("build")
            };
            assert_eq!(why, "build");
        }

        // an unsigned-by-us catalog is rejected like an unreachable one
        let foreign = serde_json::to_vec(&vec![sign(&mlx, "test", &key(9))]).unwrap();
        let base = serve(HashMap::from([("/c.json".to_string(), foreign)]));
        let url = format!("{base}/c.json");
        let cfg = Catalog {
            url: &url,
            allow_loopback_http: true,
        };
        assert!(matches!(
            check(&models, &rt, &cfg, "d".into()),
            SetupView::Offline { .. }
        ));
        let _ = std::fs::remove_dir_all(dir);
    }
}
