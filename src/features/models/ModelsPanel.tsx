import { useCallback, useEffect, useRef, useState } from "react";
import { ConfirmDelete } from "../../components/ConfirmDelete";
import { useI18n } from "../../i18n";
import { hostCode, hostText, reasonText } from "../../i18n/host";
import { api, errorText, onBackend, onModelOp } from "../../lib/api";
import { friendlyName } from "../../lib/format";
import type { BackendState, ModelOp, ModelsOverview, PendingInstall } from "../../types/protocol";
import { SetupScreen } from "../setup/SetupScreen";
import { cardStatus, onboardStage } from "./onboard";

interface Props {
  backend: BackendState | null;
  onOpenModels: () => void;
}

export function ModelsPanel({ backend, onOpenModels }: Props) {
  const { t, number } = useI18n();
  const mb = (n: number) => (n >= 1 << 30 ? `${number(n / (1 << 30), { maximumFractionDigits: 2 })} GiB` : `${number(n / (1 << 20), { maximumFractionDigits: 1 })} MiB`);
  const [o, setO] = useState<ModelsOverview | null>(null);
  const [op, setOp] = useState<ModelOp | null>(null);
  const [pending, setPending] = useState<PendingInstall | null>(null);
  const [accepted, setAccepted] = useState<string[]>([]);
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
    });
  };
  const confirmInstall = () => act(async () => {
    if (!pending) return;
    await api.modelsInstall(pending.token, accepted);
    closePending();
  });

  const running = op?.state === "running";
  const required = pending?.licenses.filter((l) => l.acceptance_required && !l.accepted) ?? [];
  const canInstall = required.every((l) => accepted.includes(l.id));
  const stage = onboardStage(op, backend, t);
  const inUse = backend && !backend.info.mock ? o?.installed.find((m) => backend.info.label.startsWith(m.id)) : undefined;
  const hasPrevious = o?.installed.some((m) => m.previous) ?? false;

  return (
    <section>
      {error && <p className="error" role="alert">{error}</p>}
      {note && <p className="muted" role="status">{note}</p>}

      <article className="card summary">
        {inUse ? (
          <strong>{t("models.in_use", { name: friendlyName(inUse.id) })}</strong>
        ) : (
          <>
            <strong>{t("models.none")}</strong>
            {backend && <p className="muted">{reasonText(backend.info.reason, t)}</p>}
          </>
        )}
      </article>

      {pending && (
        <article className="card" aria-labelledby="licence-title" onKeyDown={(e) => e.key === "Escape" && closePending()}>
          <h2 id="licence-title" ref={heading} tabIndex={-1}>{t("models.licence_title", { name: friendlyName(pending.id) })}</h2>
          <p className="muted small">
            {pending.family} · {pending.backend} · {pending.precision} · {mb(pending.download_bytes)}
          </p>
          {pending.licenses.filter((l) => l.acceptance_required && !l.accepted).map((l) => (
            <label key={l.id} className="block">
              <input type="checkbox" checked={accepted.includes(l.id)} onChange={(e) => setAccepted((a) => (e.target.checked ? [...a, l.id] : a.filter((x) => x !== l.id)))} /> {t("models.accept", { name: `${l.name}${l.spdx ? ` (${l.spdx})` : ""}` })}
              <div className="muted small">{l.url}{l.notes ? ` · ${l.notes}` : ""}</div>
            </label>
          ))}
          {pending.licenses.some((l) => l.acceptance_required && l.accepted) && (
            <p className="muted small">{t("models.accepted_before", { names: pending.licenses.filter((l) => l.acceptance_required && l.accepted).map((l) => l.name).join(", ") })}</p>
          )}
          {pending.licenses.some((l) => !l.acceptance_required) && (
            <>
              <p className="muted small">{t("models.included")}</p>
              <ul className="small">{pending.licenses.filter((l) => !l.acceptance_required).map((l) => <li key={l.id}>{l.name}{l.spdx ? ` (${l.spdx})` : ""} · {l.url}</li>)}</ul>
            </>
          )}
          <div className="row">
            <button className="primary" disabled={!canInstall || running} onClick={confirmInstall}>{required.length ? t("models.accept_add") : t("models.add_model")}</button>
            <button onClick={closePending}>{t("common.cancel")}</button>
          </div>
        </article>
      )}

      {op && (o?.installed.length ?? 0) > 0 && (
        <article className={`card stage ${stage?.phase ?? ""}`} aria-live="polite">
          <div className="row">
            <strong className="ellipsis" title={op.name}>{friendlyName(op.name)}</strong>
            {running && <button onClick={() => act(api.modelsCancel)}>{t("common.cancel")}</button>}
          </div>
          {running && (op.bytes_total > 0 ? <progress max={op.bytes_total} value={op.bytes_done} /> : <progress />)}
          {stage && <p className={stage.phase === "problem" ? "error" : stage.phase === "ready" ? "ok" : "muted"}>{stage.text}</p>}
        </article>
      )}

      {o && o.installed.length === 0 && <SetupScreen backend={backend} onOpenModels={onOpenModels} />}
      {o?.installed.map((m) => {
        const inUseHere = !!backend && !backend.info.mock && backend.info.label.startsWith(m.id);
        const status = cardStatus(m, backend, op, t);
        return (
          <article key={m.id} className="card model">
            <div className="row between">
              <strong className="ellipsis" title={m.id}>{friendlyName(m.id)}</strong>
              {inUseHere && <span className="pill done">{t("models.badge_in_use")}</span>}
              {m.previous && <span className="pill">{t("models.badge_previous")}</span>}
              <span className="spacer" />
              {inUseHere ? (
                <button disabled={running} onClick={() => act(() => api.modelsProbe(m.id))}>{t("models.test_again")}</button>
              ) : (
                <button className="primary" disabled={m.active || running} onClick={() => act(() => api.modelsActivate(m.id))}>{t("models.use")}</button>
              )}
            </div>
            <p className={hostCode(backend?.info.reason) === "backend.missing_engine" || /test_failed/.test(hostCode(backend?.info.reason) ?? "") ? (m.active ? "error small" : "muted small") : "muted small"}>{status}</p>
            <details>
              <summary>{t("models.details")}</summary>
              <dl className="kv small">
                <dt>{t("models.d_id")}</dt><dd>{m.id}</dd>
                <dt>{t("models.d_engine")}</dt><dd>{m.backend} · {m.precision}</dd>
                <dt>{t("models.d_size")}</dt><dd>{mb(m.size_bytes)}</dd>
                <dt>{t("models.d_serial")}</dt><dd>{m.serial} / {m.revision}</dd>
                <dt>{t("models.d_validation")}</dt><dd>{m.compat_status}. {m.compat_notes}</dd>
                {m.licenses.length > 0 && (<><dt>{t("models.d_licences")}</dt><dd>{m.licenses.join(", ")}</dd></>)}
              </dl>
              <div className="row">
                <button disabled={running} onClick={() => act(() => api.modelsVerify(m.id), t("models.verified", { name: friendlyName(m.id) }))}>{t("models.verify")}</button>
                {inUseHere && hasPrevious && <button disabled={running} onClick={() => act(() => api.modelsRollback().then((id) => setNote(t("models.rolled_back", { name: friendlyName(id) }))))}>{t("models.rollback")}</button>}
                {removing === m.id ? (
                  <ConfirmDelete question={t("models.remove_q", { name: friendlyName(m.id) })} action={t("common.remove")} onCancel={() => setRemoving(null)} onConfirm={() => { setRemoving(null); void act(() => api.modelsRemove(m.id), t("models.removed", { name: friendlyName(m.id) })); }} />
                ) : (
                  <button disabled={m.active || m.previous || running} onClick={() => setRemoving(m.id)}>{t("models.remove")}</button>
                )}
              </div>
            </details>
          </article>
        );
      })}
      {o?.unreadable.map(([id, why]) => <p key={id} className="error">{id}: {why}</p>)}

      <details className="card">
        <summary>{t("models.advanced")}</summary>
        <h3>{t("models.add_title")}</h3>
      <div className="row">
        <button className="primary" onClick={() => begin(() => api.modelsBeginImport("folder"))} disabled={running || !o?.registry_available}>{t("models.add_folder")}</button>
        <button onClick={() => begin(() => api.modelsBeginImport("zip"))} disabled={running || !o?.registry_available}>{t("models.add_zip")}</button>
        <button onClick={() => begin(api.modelsBeginInstall)} disabled={running || !o?.registry_available}>{t("models.install_manifest")}</button>
      </div>


        {o?.capabilities && (
          <p className="muted small">
            {o.capabilities.apple_chip
              ? t("models.computer_chip", { os: o.capabilities.os, arch: o.capabilities.arch, chip: o.capabilities.apple_chip, ram: mb(o.capabilities.ram_bytes) })
              : t("models.computer", { os: o.capabilities.os, arch: o.capabilities.arch, ram: mb(o.capabilities.ram_bytes) })}
          </p>
        )}
        {o?.selection && (
          <>
            <p className="muted small">{t("models.selection", { reason: reasonText(o.selection.reason, t) })}</p>
            <ul className="small">{o.selection.rejected.map(([id, why]) => <li key={id}>{id}: {hostText(why, t)}</li>)}</ul>
          </>
        )}
        {o && <p className="muted small">{hostText(o.note, t)}</p>}
        {o && <p className="muted small">{t("models.registry", { path: o.root })}</p>}
      </details>
    </section>
  );
}
