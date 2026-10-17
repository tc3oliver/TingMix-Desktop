import { useCallback, useEffect, useState } from "react";
import { api, errorText, onModelOp } from "../../lib/api";
import type { ModelOp, ModelsOverview, PendingInstall } from "../../types/protocol";

const mb = (n: number) => (n >= 1 << 30 ? `${(n / (1 << 30)).toFixed(2)} GiB` : `${(n / (1 << 20)).toFixed(1)} MiB`);

export function ModelsPanel() {
  const [o, setO] = useState<ModelsOverview | null>(null);
  const [op, setOp] = useState<ModelOp | null>(null);
  const [pending, setPending] = useState<PendingInstall | null>(null);
  const [accepted, setAccepted] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const reload = useCallback(() => api.modelsOverview().then((v) => { setO(v); setOp(v.op); }).catch((e) => setError(errorText(e))), []);
  useEffect(() => {
    void reload();
    let alive = true;
    let off: (() => void) | undefined;
    onModelOp((v) => {
      setOp(v);
      if (v.state !== "running") void reload();
    }).then((u) => (alive ? (off = u) : u()));
    return () => {
      alive = false;
      off?.();
    };
  }, [reload]);

  const act = async (f: () => Promise<unknown>, ok?: string) => {
    setError(null);
    setNote(null);
    try {
      await f();
      if (ok) setNote(ok);
    } catch (e) {
      setError(errorText(e));
    }
    await reload();
  };

  const begin = (f: () => Promise<PendingInstall | null>) => act(async () => {
    const p = await f();
    setPending(p);
    setAccepted(p ? p.licenses.filter((l) => l.accepted).map((l) => l.id) : []);
  });
  const confirmInstall = () => act(async () => {
    if (!pending) return;
    await api.modelsInstall(pending.token, accepted);
    setPending(null);
  });

  const running = op?.state === "running";
  const needed = pending?.licenses.filter((l) => l.acceptance_required && !l.accepted) ?? [];
  const canInstall = needed.every((l) => accepted.includes(l.id));

  return (
    <section>
      {error && <p className="error" role="alert">{error}</p>}
      {note && <p className="muted">{note}</p>}
      {o && <p className="muted">{o.note}</p>}
      {o?.capabilities && (
        <p className="muted small">
          This computer: {o.capabilities.os}/{o.capabilities.arch}{o.capabilities.apple_chip ? ` · ${o.capabilities.apple_chip}` : ""} · {mb(o.capabilities.ram_bytes)} RAM
        </p>
      )}

      <div className="row toolbar">
        <button onClick={() => begin(api.modelsBeginInstall)} disabled={running || !o?.registry_available}>Install from signed manifest…</button>
        <button onClick={() => begin(() => api.modelsBeginImport("folder"))} disabled={running || !o?.registry_available}>Import folder…</button>
        <button onClick={() => begin(() => api.modelsBeginImport("zip"))} disabled={running || !o?.registry_available}>Import zip…</button>
        <button onClick={() => act(() => api.modelsRollback().then((id) => setNote(`Rolled back to ${id}`)))} disabled={running || !o?.installed.some((m) => m.previous)}>Roll back</button>
      </div>

      {pending && (
        <article className="card" aria-label="Confirm install">
          <strong>{pending.id}</strong>
          <p className="muted small">
            {pending.family} · {pending.backend} · {pending.precision} · {mb(pending.download_bytes)} to download · status {pending.compat_status}
          </p>
          {pending.licenses.map((l) => (
            <label key={l.id} className="block">
              <input
                type="checkbox"
                disabled={!l.acceptance_required || l.accepted}
                checked={!l.acceptance_required || accepted.includes(l.id)}
                onChange={(e) => setAccepted((a) => (e.target.checked ? [...a, l.id] : a.filter((x) => x !== l.id)))}
              />{" "}
              {l.name}{l.spdx ? ` (${l.spdx})` : ""}{l.acceptance_required ? (l.accepted ? " — already accepted" : " — acceptance required") : ""}
              <div className="muted small">{l.url}{l.notes ? ` · ${l.notes}` : ""}</div>
            </label>
          ))}
          <div className="row">
            <button className="primary" disabled={!canInstall || running} onClick={confirmInstall}>Accept licences and install</button>
            <button onClick={() => setPending(null)}>Cancel</button>
          </div>
        </article>
      )}

      {op && (
        <article className="card" aria-live="polite">
          <div className="row">
            <strong>{op.kind === "download" ? "Download" : op.kind === "probe" ? "Probe" : "Import"}: {op.name}</strong>
            <span className={`pill ${op.state === "running" ? "running" : op.state === "done" ? "done" : "failed"}`}>{op.state}</span>
            {running && <button onClick={() => act(api.modelsCancel)}>Cancel</button>}
          </div>
          {running && (op.bytes_total > 0 ? <progress max={op.bytes_total} value={op.bytes_done} /> : <progress />)}
          {running && op.bytes_total > 0 && <span className="muted small">{mb(op.bytes_done)} of {mb(op.bytes_total)}</span>}
          {op.message && <p className={op.state === "failed" ? "error" : "muted"}>{op.message}</p>}
        </article>
      )}

      <h3>Installed versions</h3>
      {o && o.installed.length === 0 && <p className="muted">No model installed.</p>}
      {o?.installed.map((m) => (
        <article key={m.id} className="card">
          <div className="row">
            <strong>{m.id}</strong>
            {m.active && <span className="pill done">active</span>}
            {m.previous && <span className="pill">previous</span>}
            <span className="muted small">{m.family} · {m.backend} · {m.precision} · {mb(m.size_bytes)} · serial {m.serial} · rev {m.revision.slice(0, 8)}</span>
          </div>
          <p className="muted small">Status: {m.compat_status}. {m.compat_notes}</p>
          {m.licenses.length > 0 && <p className="muted small">Licences: {m.licenses.join(", ")}</p>}
          <div className="row">
            <button disabled={running} onClick={() => act(() => api.modelsProbe(m.id), `Compatibility probe of ${m.id} finished`)}>Run compatibility probe</button>
            <button disabled={m.active || running} onClick={() => act(() => api.modelsActivate(m.id), `Activated ${m.id}`)}>Make active</button>
            <button disabled={running} onClick={() => act(() => api.modelsVerify(m.id), `${m.id}: every file matches its signed hash`)}>Verify files</button>
            <button disabled={m.active || m.previous || running} onClick={() => act(() => api.modelsRemove(m.id), `Removed ${m.id}`)}>Remove</button>
          </div>
        </article>
      ))}
      {o?.unreadable.map(([id, why]) => <p key={id} className="error">{id}: {why}</p>)}
      {o?.selection && (
        <>
          <h3>Runtime selection</h3>
          <p className="muted">{o.selection.reason}</p>
          <ul className="small">{o.selection.rejected.map(([id, why]) => <li key={id}>{id}: {why}</li>)}</ul>
        </>
      )}
      {o && <p className="muted small">Registry folder: {o.root}</p>}
    </section>
  );
}
