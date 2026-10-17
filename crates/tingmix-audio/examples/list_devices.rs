//! `cargo run -p tingmix-audio --example list_devices`
fn main() {
    match tingmix_audio::list_input_devices() {
        Ok(d) if d.is_empty() => println!("no input devices"),
        Ok(devices) => {
            for d in devices {
                let star = if d.is_default { "*" } else { " " };
                println!(
                    "{star} {}\n    id={} {} Hz, {} ch",
                    d.name, d.id, d.sample_rate, d.channels
                );
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
