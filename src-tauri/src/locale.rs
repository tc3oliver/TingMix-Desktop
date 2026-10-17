//! Host-side strings the webview cannot translate: native dialog titles and filter names, tray items. Everything else the
//! host says reaches the UI as a coded message (`coded.rs`). Adding a language: add a variant to `Lang` and a column here.

use crate::dto::Settings;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    En,
    ZhTw,
}

impl Lang {
    /// `language` is the saved setting (`auto`, `en`, `zh-TW`); `auto` follows the OS language.
    pub fn resolve(language: &str, system: &[String]) -> Self {
        match language {
            "en" => Lang::En,
            "zh-TW" => Lang::ZhTw,
            _ => system
                .iter()
                .find_map(|l| Self::from_tag(l))
                .unwrap_or(Lang::En),
        }
    }

    /// Traditional Chinese for `zh-TW`, `zh-HK`, `zh-MO` and any `zh-Hant*`; the first language the user ranks decides.
    fn from_tag(tag: &str) -> Option<Self> {
        let t = tag.to_ascii_lowercase().replace('_', "-");
        if t == "zh-tw" || t == "zh-hk" || t == "zh-mo" || t.starts_with("zh-hant") {
            Some(Lang::ZhTw)
        } else {
            Some(Lang::En)
        }
    }

    pub fn current(settings: &Settings) -> Self {
        let system: Vec<String> = sys_locale::get_locales().collect();
        Self::resolve(&settings.language, &system)
    }
}

pub enum Text {
    PickAudio,
    FilterAudio,
    ExportTitle,
    PickModelPackage,
    FilterModelArchive,
    PickManifest,
    FilterManifest,
    TrayShow,
    TrayQuit,
}

pub fn tr(lang: Lang, t: Text) -> &'static str {
    use Lang::*;
    use Text::*;
    match (t, lang) {
        (PickAudio, En) => "Choose an audio file",
        (PickAudio, ZhTw) => "選擇音訊檔案",
        (FilterAudio, En) => "Audio",
        (FilterAudio, ZhTw) => "音訊",
        (ExportTitle, En) => "Export transcript",
        (ExportTitle, ZhTw) => "匯出逐字稿",
        (PickModelPackage, En) => "Choose a model package",
        (PickModelPackage, ZhTw) => "選擇模型套件",
        (FilterModelArchive, En) => "Model archive",
        (FilterModelArchive, ZhTw) => "模型壓縮檔",
        (PickManifest, En) => "Choose a signed model manifest",
        (PickManifest, ZhTw) => "選擇已簽署的模型清單",
        (FilterManifest, En) => "Signed manifest",
        (FilterManifest, ZhTw) => "已簽署的清單",
        (TrayShow, En) => "Show TingMix",
        (TrayShow, ZhTw) => "顯示 TingMix",
        (TrayQuit, En) => "Quit",
        (TrayQuit, ZhTw) => "結束",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn explicit_choice_wins_and_auto_follows_the_first_ranked_language() {
        assert_eq!(Lang::resolve("en", &tags(&["zh-TW"])), Lang::En);
        assert_eq!(Lang::resolve("zh-TW", &tags(&["en-US"])), Lang::ZhTw);
        for tag in ["zh-TW", "zh_TW", "zh-Hant", "zh-Hant-TW", "zh-HK", "zh-MO"] {
            assert_eq!(
                Lang::resolve("auto", &tags(&[tag, "en-US"])),
                Lang::ZhTw,
                "{tag}"
            );
        }
        assert_eq!(Lang::resolve("auto", &tags(&["en-US", "zh-TW"])), Lang::En);
        assert_eq!(Lang::resolve("auto", &tags(&["zh-CN"])), Lang::En);
        assert_eq!(Lang::resolve("auto", &[]), Lang::En);
    }

    #[test]
    fn texts_differ_per_language() {
        assert_ne!(tr(Lang::En, Text::TrayQuit), tr(Lang::ZhTw, Text::TrayQuit));
    }
}
