//! Hardware capability detection. GPU vendor is reported as a hint only and is never an input to compatibility.

use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppleChip {
    pub name: String,
    /// 1 = M1 ... 4 = M4.
    pub generation: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    pub os: String,
    pub arch: String,
    pub cpu_features: Vec<String>,
    pub ram_bytes: u64,
    pub apple_chip: Option<AppleChip>,
    /// Display only (diagnostics panel). Not used by [`crate::select`].
    pub gpu_vendor_hint: Option<String>,
}

pub fn parse_apple_chip(brand: &str) -> Option<AppleChip> {
    let rest = brand.trim().strip_prefix("Apple M")?;
    let generation = rest
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()?;
    Some(AppleChip {
        name: brand.trim().to_string(),
        generation,
    })
}

fn cpu_features() -> Vec<String> {
    let mut v: Vec<&str> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        macro_rules! probe { ($($f:tt),*) => { $( if std::arch::is_x86_feature_detected!($f) { v.push($f); } )* } }
        probe!("sse4.2", "avx", "avx2", "fma", "avx512f");
    }
    #[cfg(target_arch = "aarch64")]
    {
        macro_rules! probe { ($($f:tt),*) => { $( if std::arch::is_aarch64_feature_detected!($f) { v.push($f); } )* } }
        probe!("neon", "fp16", "dotprod");
    }
    v.into_iter().map(String::from).collect()
}

impl Capabilities {
    pub fn detect() -> Self {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let apple_chip = if cfg!(target_os = "macos") {
            Command::new("sysctl")
                .args(["-n", "machdep.cpu.brand_string"])
                .output()
                .ok()
                .and_then(|o| parse_apple_chip(&String::from_utf8_lossy(&o.stdout)))
        } else {
            None
        };
        Self {
            os: std::env::consts::OS.into(),
            arch: std::env::consts::ARCH.into(),
            cpu_features: cpu_features(),
            ram_bytes: sys.total_memory(),
            gpu_vendor_hint: apple_chip.as_ref().map(|_| "Apple".to_string()),
            apple_chip,
        }
    }

    /// Stable identity of everything the selection depends on, so a change (new chip, OS, much less RAM) invalidates
    /// the persisted last-known-good choice. RAM is bucketed to whole GiB.
    pub fn fingerprint(&self) -> String {
        format!(
            "{}/{}/{}/{}/{}GiB",
            self.os,
            self.arch,
            self.cpu_features.join("+"),
            self.apple_chip
                .as_ref()
                .map(|c| c.name.as_str())
                .unwrap_or("-"),
            self.ram_bytes >> 30
        )
    }
}
