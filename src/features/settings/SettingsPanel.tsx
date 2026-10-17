import type { Settings } from "../../types/protocol";

export function SettingsPanel({ settings, onSettings }: { settings: Settings; onSettings: (s: Settings) => void }) {
  return (
    <section className="form">
      <label>
        Theme{" "}
        <select value={settings.theme} onChange={(e) => onSettings({ ...settings, theme: e.target.value as Settings["theme"] })}>
          <option value="system">System</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </label>
      <label>
        Caption size: {settings.caption_size}px{" "}
        <input type="range" min={12} max={72} value={settings.caption_size} onChange={(e) => onSettings({ ...settings, caption_size: Number(e.target.value) })} />
      </label>
      <label>
        <input type="checkbox" checked={settings.save_history} onChange={(e) => onSettings({ ...settings, save_history: e.target.checked })} /> Save finished transcripts to history
      </label>
      <p className="muted small">Settings are stored locally and applied immediately.</p>
    </section>
  );
}
