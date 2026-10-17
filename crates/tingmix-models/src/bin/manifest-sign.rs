//! Publisher tool: generate an Ed25519 key, sign an unsigned manifest, verify a signed one.
//!
//!   manifest-sign keygen <key-file>                         new key, file mode 600, prints the PUBLIC key only
//!   manifest-sign pubkey <key-file>                         prints the public key (hex)
//!   manifest-sign sign <key-file> <key-id> <in.json> <out>  sign a manifest
//!   manifest-sign verify <signed.json>                      verify against the keys embedded in this build
//!
//! The private key is read from and written to `<key-file>` only; it is never printed or accepted as an argument.

use std::fs;
use std::io::Write;
use std::process::ExitCode;

use ed25519_dalek::SigningKey;
use tingmix_models::manifest::{hex, sign, unhex, Manifest};
use tingmix_models::Verifier;

fn load_key(path: &str) -> Result<SigningKey, String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(path)
            .map_err(|e| format!("{path}: {e}"))?
            .permissions()
            .mode()
            & 0o777;
        if mode != 0o600 {
            return Err(format!("{path}: key file mode is {mode:o}, must be 600"));
        }
    }
    let seed = unhex(
        fs::read_to_string(path)
            .map_err(|e| format!("{path}: {e}"))?
            .trim(),
    )
    .and_then(|b| <[u8; 32]>::try_from(b).ok())
    .ok_or("key file is not 64 hex chars")?;
    Ok(SigningKey::from_bytes(&seed))
}

fn run(args: &[String]) -> Result<(), String> {
    let a: Vec<&str> = args.iter().map(String::as_str).collect();
    match a.as_slice() {
        ["keygen", path] => {
            let mut seed = [0u8; 32];
            getrandom::fill(&mut seed).map_err(|e| e.to_string())?;
            let mut o = fs::OpenOptions::new();
            o.write(true).create_new(true);
            #[cfg(unix)]
            std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
            o.open(path)
                .map_err(|e| format!("{path}: {e}"))?
                .write_all(format!("{}\n", hex(&seed)).as_bytes())
                .map_err(|e| e.to_string())?;
            println!(
                "public key: {}",
                hex(&SigningKey::from_bytes(&seed).verifying_key().to_bytes())
            );
            Ok(())
        }
        ["pubkey", path] => {
            println!("{}", hex(&load_key(path)?.verifying_key().to_bytes()));
            Ok(())
        }
        ["sign", key, key_id, input, out] => {
            let m: Manifest =
                serde_json::from_slice(&fs::read(input).map_err(|e| format!("{input}: {e}"))?)
                    .map_err(|e| format!("{input}: {e}"))?;
            m.validate(false).map_err(|e| e.to_string())?;
            let signed = sign(&m, key_id, &load_key(key)?);
            fs::write(out, serde_json::to_vec_pretty(&signed).unwrap())
                .map_err(|e| format!("{out}: {e}"))?;
            println!(
                "signed {} ({} files, {} bytes) with key {key_id}",
                m.id,
                m.files.len(),
                m.download_size()
            );
            Ok(())
        }
        ["verify", input] => {
            let v = Verifier::embedded()
                .parse(&fs::read(input).map_err(|e| format!("{input}: {e}"))?)
                .map_err(|e| e.to_string())?;
            v.manifest().validate(false).map_err(|e| e.to_string())?;
            println!("ok: {} signed by {}", v.manifest().id, v.signed().key_id);
            Ok(())
        }
        _ => Err("usage: manifest-sign keygen|pubkey|sign|verify ... (see source header)".into()),
    }
}

fn main() -> ExitCode {
    match run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
