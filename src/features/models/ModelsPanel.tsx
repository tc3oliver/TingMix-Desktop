import { useCallback, useEffect, useRef, useState } from "react";
import { ConfirmDelete } from "../../components/ConfirmDelete";
import { api, errorText, onBackend, onModelOp } from "../../lib/api";
import { friendlyName } from "../../lib/format";
import type { BackendState, ModelOp, ModelsOverview, PendingInstall, Settings } from "../../types/protocol";
import { cardStatus, onboardStage } from "./onboard";

const mb = (n: number) => (n >= 1 << 30 ? `${(n / (1 << 30)).toFixed(2)} GiB` : `${(n / (1 << 20)).toFixed(1)} MiB`);

interface Props {
  backend: BackendState | null;
  settings: Settings;
  onSettings: (s: Settings) => void;
}

export function ModelsPanel({ backend, settings, onSettings }: Props) {
  const [o, setO] = useState<ModelsOverview | null>(null);
  const [op, setOp] = useState<ModelOp | null>(null);
  const [pending, setPending] = useState<PendingInstall | null>(null);
  const [accepted, setAccepted] = useState<string[]>([]);
  const [enableExperimental, setEnableExperimental] = useState(false);
  const [removing, setRemoving] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const opener = useRef<HTMLElement | null>(null);

  const reload = useCallback(() => api.modelsOverview().then((v) => setO(v)).catch((e) => setError(errorText(e))), []);
  useEffect(() => {
    void reload();
    let alive = true;
    const offs: (() => void)[] = [];
    const keep = (p: Promise<() => void>) => p.then((u) => (alive ? offs.push(u) : u()));
    keep(onModelOp((v) => setOp(v)));
    // activation, test results, rollback and settings all arrive as one backend event: the list follows it directly
    keep(onBackend(() => void reload()));
    void api.modelsOverview().then((v) => alive && setOp(v.op));
    return () => {
      alive = false;
      offs.forEach((f) => f());
    };
  }, [reload]);

  // the licence card takes focus when it opens and gives it back when it closes
  useEffect(() => {
    if (pending) heading.current?.focus();
  }, [pending?.token]); // eslint-disable-line react-hooks/exhaustive-deps
  const closePending = () => {
    setPending(null);
    opener.current?.focus();
  };

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

  const begin = (f: () => Promise<PendingInstall | null>) => {
    opener.current = document.activeElement as HTMLElement | null;
    return act(async () => {
      const p = await f();
      setPending(p);
      setAccepted([]);
      setEnableExperimental(false);
    });
  };
  const confirmInstall = () => act(async () => {
    if (!pending) return;
    await api.modelsInstall(pending.token, accepted, enableExperimental);
    closePending();
  });

  const running = op?.state === "running";
  const required = pending?.licenses.filter((l) => l.acceptance_required && !l.accepted) ?? [];
  const canInstall = required.every((l) => accepted.includes(l.id));
  const stage = onboardStage(op, backend);
  const inUse = backend && !backend.info.mock ? o?.installed.find((m) => backend.info.label.startsWith(m.id)) : undefined;
  const hasPrevious = o?.installed.some((m) => m.previous) ?? false;

  return (
    <section>
      {error && <p className="error" role="alert">{error}</p>}
      {note && <p className="muted" role="status">{note}</p>}

      <article className="card summary">
        {inUse ? (
          <strong>In use: {friendlyName(inUse.id)}</strong>
        ) : (
          <>
            <strong>No model in use — demo mode</strong>
            {backend && <p className="muted">{backend.info.reason}</p>}
          </>
        )}
      </article>

      <div className="row toolbar">
        <button className="primary" onClick={() => begin(() => api.modelsBeginImport("folder"))} disabled={running || !o?.registry_available}>Add model folder…</button>
        <button onClick={() => begin(() => api.modelsBeginImport("zip"))} disabled={running || !o?.registry_available}>Add .zip…</button>
        <button onClick={() => begin(api.modelsBeginInstall)} disabled={running || !o?.registry_available}>Install from signed manifest…</button>
      </div>

      {pending && (
        <article className="card" aria-labelledby="licence-title" onKeyDown={(e) => e.key === "Escape" && closePending()}>
          <h2 id="licence-title" ref={heading} tabIndex={-1}>Review licences for {friendlyName(pending.id)}</h2>
          <p className="muted small">
            {pending.family} · {pending.backend} · {pending.precision} · {mb(pending.download_bytes)}
          </p>
          {pending.licenses.filter((l) => l.acceptance_required && !l.accepted).map((l) => (
            <label key={l.id} className="block">
              <input type="checkbox" checked={accepted.includes(l.id)} onChange={(e) => setAccepted((a) => (e.target.checked ? [...a, l.id] : a.filter((x) => x !== l.id)))} /> I accept {l.name}{l.spdx ? ` (${l.spdx})` : ""}
              <div className="muted small">{l.url}{l.notes ? ` · ${l.notes}` : ""}</div>
            </label>
          ))}
          {pending.licenses.some((l) => l.acceptance_required && l.accepted) && (
            <p className="muted small">Accepted before: {pending.licenses.filter((l) => l.acceptance_required && l.accepted).map((l) => l.name).join(", ")}</p>
          )}
          {pending.licenses.some((l) => !l.acceptance_required) && (
            <>
              <p className="muted small">Included licences</p>
              <ul className="small">{pending.licenses.filter((l) => !l.acceptance_required).map((l) => <li key={l.id}>{l.name}{l.spdx ? ` (${l.spdx})` : ""} · {l.url}</li>)}</ul>
            </>
          )}
          {pending.experimental && (
            <label className="block">
              <input type="checkbox" checked={enableExperimental} onChange={(e) => setEnableExperimental(e.target.checked)} /> Enable experimental engines so this model can be used
              <div className="muted small">Its output has not been validated against the reference. You can turn this off later in Settings.</div>
            </label>
          )}
          <p className="muted small">After adding, TingMix makes this the model in use (if none is) and tests it on this Mac.</p>
          <div className="row">
            <button className="primary" disabled={!canInstall || running} onClick={confirmInstall}>{required.length ? "Accept and add" : "Add model"}</button>
            <button onClick={closePending}>Cancel</button>
          </div>
        </article>
      )}

      {op && (
        <article className={`card stage ${stage?.phase ?? ""}`} aria-live="polite">
          <div className="row">
            <strong className="ellipsis" title={op.name}>{friendlyName(op.name)}</strong>
            {running && <button onClick={() => act(api.modelsCancel)}>Cancel</button>}
          </div>
          {running && (op.bytes_total > 0 ? <progress max={op.bytes_total} value={op.bytes_done} /> : <progress />)}
          {stage && <p className={stage.phase === "problem" ? "error" : stage.phase === "ready" ? "ok" : "muted"}>{stage.text}</p>}
        </article>
      )}

      {o && o.installed.length === 0 && (
        <article className="card">
          <h2>Get started</h2>
          <ol>
            <li>Get a TingMix model package (a folder or .zip).</li>
            <li>Add it here — TingMix checks and tests it automatically.</li>
            <li>Start transcribing.</li>
          </ol>
        </article>
      )}
      {o?.installed.map((m) => {
        const inUseHere = !!backend && !backend.info.mock && backend.info.label.startsWith(m.id);
        const status = cardStatus(m, backend, op);
        const experimental = status.startsWith("Experimental") && !settings.experimental_backends;
        return (
          <article key={m.id} className="card model">
            <div className="row between">
              <strong className="ellipsis" title={m.id}>{friendlyName(m.id)}</strong>
              {inUseHere && <span className="pill done">In use</span>}
              {m.previous && <span className="pill">Previous version</span>}
              <span className="spacer" />
              {inUseHere ? (
                <button disabled={running} onClick={() => act(() => api.modelsProbe(m.id))}>Test again</button>
              ) : (
                <button className="primary" disabled={m.active || running} onClick={() => act(() => api.modelsActivate(m.id))}>Use this model</button>
              )}
            </div>
            <p className={status.startsWith("Test failed") || status.startsWith("Not supported") ? "error small" : "muted small"}>{status}</p>
            {experimental && <button onClick={() => onSettings({ ...settings, experimental_backends: true })}>Turn on experimental engines</button>}
            <details>
              <summary>Technical details</summary>
              <dl className="kv small">
                <dt>Id</dt><dd>{m.id}</dd>
                <dt>Engine</dt><dd>{m.backend} · {m.precision}</dd>
                <dt>Size</dt><dd>{mb(m.size_bytes)}</dd>
                <dt>Serial / revision</dt><dd>{m.serial} / {m.revision}</dd>
                <dt>Validation status</dt><dd>{m.compat_status}. {m.compat_notes}</dd>
                {m.licenses.length > 0 && (<><dt>Licences</dt><dd>{m.licenses.join(", ")}</dd></>)}
              </dl>
              <div className="row">
                <button disabled={running} onClick={() => act(() => api.modelsVerify(m.id), `Every file of ${friendlyName(m.id)} matches its signed hash.`)}>Verify files</button>
                {inUseHere && hasPrevious && <button disabled={running} onClick={() => act(() => api.modelsRollback().then((id) => setNote(`Rolled back to ${friendlyName(id)}.`)))}>Roll back to previous version</button>}
                {removing === m.id ? (
                  <ConfirmDelete question={`Remove “${friendlyName(m.id)}”? Its files are deleted from this computer.`} action="Remove" onCancel={() => setRemoving(null)} onConfirm={() => { setRemoving(null); void act(() => api.modelsRemove(m.id), `Removed ${friendlyName(m.id)}.`); }} />
                ) : (
                  <button disabled={m.active || m.previous || running} onClick={() => setRemoving(m.id)}>Remove…</button>
                )}
              </div>
            </details>
          </article>
        );
      })}
      {o?.unreadable.map(([id, why]) => <p key={id} className="error">{id}: {why}</p>)}

      <details className="card">
        <summary>Advanced</summary>
        {o?.capabilities && (
          <p className="muted small">
            This computer: {o.capabilities.os}/{o.capabilities.arch}{o.capabilities.apple_chip ? ` · ${o.capabilities.apple_chip}` : ""} · {mb(o.capabilities.ram_bytes)} RAM
          </p>
        )}
        {o?.selection && (
          <>
            <p className="muted small">Engine selection: {o.selection.reason}</p>
            <ul className="small">{o.selection.rejected.map(([id, why]) => <li key={id}>{id}: {why}</li>)}</ul>
          </>
        )}
        {o && <p className="muted small">{o.note}</p>}
        {o && <p className="muted small">Registry folder: {o.root}</p>}
      </details>
    </section>
  );
}
