use cpal::traits::{DeviceTrait, HostTrait};

use crate::AudioError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputDevice {
    /// Stable identifier; pass it to [`crate::start_capture`] and persist it in settings.
    pub id: String,
    pub name: String,
    pub is_default: bool,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Input devices of the default audio host. An empty list is a valid answer (e.g. a desktop Mac
/// without a microphone); a device whose default config cannot be read is skipped.
pub fn list_input_devices() -> Result<Vec<InputDevice>, AudioError> {
    let host = cpal::default_host();
    let default_id = host.default_input_device().and_then(|d| d.id().ok());
    let mut out = Vec::new();
    for d in host.input_devices()? {
        let (Ok(id), Ok(cfg)) = (d.id(), d.default_input_config()) else {
            continue;
        };
        let name = d
            .description()
            .map_or_else(|_| id.to_string(), |x| x.name().to_owned());
        out.push(InputDevice {
            is_default: default_id.as_ref() == Some(&id),
            id: id.to_string(),
            name,
            sample_rate: cfg.sample_rate(),
            channels: cfg.channels(),
        });
    }
    Ok(out)
}
