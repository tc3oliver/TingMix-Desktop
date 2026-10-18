//! Opens an https address in the default browser with the OS opener. The caller has already checked the address against an
//! allow-list; the address is passed as a single argument, never through a shell.

use std::process::Command;

pub fn open(url: &str) -> std::io::Result<()> {
    debug_assert!(url.starts_with("https://"));
    #[cfg(target_os = "macos")]
    let status = Command::new("/usr/bin/open").arg(url).status()?;
    #[cfg(target_os = "windows")]
    // rundll32 takes the address as one argument; `cmd /c start` would parse `&` and friends
    let status = Command::new("rundll32")
        .args(["url.dll,FileProtocolHandler", url])
        .status()?;
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let status = Command::new("xdg-open").arg(url).status()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "the opener exited with {status}"
        )))
    }
}
