import { useCallback, useEffect, useRef, useState } from "react";
import { useI18n } from "../../i18n";
import { hostText, reasonText } from "../../i18n/host";
import { api, errorText, onBackend, onModelOp } from "../../lib/api";
import type { BackendState, ModelOp, SetupView } from "../../types/protocol";
import { etaSeconds, etaText, pushSample, type RateSample } from "../models/onboard";

interface Props {
  backend: BackendState | null;
  onOpenModels: () => void;
  /** Reports each result of the host's setup check, so the shell can tell a first run from a broken install. */
  onView?: (v: SetupView) => void;
  /** First run: the whole window, with the one-line promise. */
  first?: boolean;
}

/** Size as "1.3 GB" in the UI language (decimal gigabytes, what download sites show). */
export function gigabytes(bytes: number, number: (n: number, o?: Intl.NumberFormatOptions) => string): string {
  return bytes >= 1e9 ? `${number(bytes / 1e9, { maximumFractionDigits: 1 })} GB` : `${number(Math.max(1, bytes / 1e6), { maximumFractionDigits: 0 })} MB`;
}

/**
 * The first-run screen: one download, then everything else happens by itself (signature check, activation and a test on
 * this computer). The host does the work; this screen follows the model operation and backend events.
 */
export function SetupScreen({ backend, onOpenModels, onView, first = false }: Props) {
  const { t, n, number } = useI18n();
  const [view, setView] = useState<SetupView | null>(null);
  const [op, setOp] = useState<ModelOp | null>(null);
  const [accepted, setAccepted] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [slow, setSlow] = useState(false); // "checking" shows only after 300 ms, so a quick check never flashes
  const samples = useRef<RateSample[]>([]);
  const [eta, setEta] = useState<number | null>(null);

  const check = useCallback((): Promise<SetupView | null> => {
    setView(null);
    return api.setupCheck().then((v) => {
      setView(v);
      onView?.(v);
      // licences accepted on an earlier attempt stay accepted: Resume must not ask again
      if (v.state === "download") setAccepted((a) => [...new Set([...a, ...v.pending.licenses.filter((l) => l.accepted).map((l) => l.id)])]);
      return v;
    }).catch((e) => {
      setError(errorText(e));
      return null;
    });
  }, [onView]);

  useEffect(() => {
    const timer = window.setTimeout(() => setSlow(true), 300);
    void check();
    let alive = true;
    const offs: (() => void)[] = [];
    const keep = (p: Promise<() => void>) => p.then((u) => (alive ? offs.push(u) : u()));
    keep(onModelOp((o) => setOp(o)));
    keep(onBackend(() => void check())); // a finished test changes what is needed
    void api.modelsOverview().then((o) => alive && setOp(o.op));
    return () => {
      alive = false;
      window.clearTimeout(timer);
      offs.forEach((f) => f());
    };
  }, [check]);

  useEffect(() => {
    if (op?.state === "running" && op.kind === "download") {
      samples.current = pushSample(samples.current, { at: Date.now(), bytes: op.bytes_done });
      setEta(etaSeconds(samples.current, op.bytes_total));
    } else {
      samples.current = [];
      setEta(null);
    }
  }, [op]);

  const download = async (token: string, ids: string[] = accepted) => {
    setError(null);
    try {
      await api.modelsInstall(token, ids);
    } catch (e) {
      setError(errorText(e));
    }
  };
  /** One click: re-check, then start again at once when every required licence was already accepted. */
  const resume = async () => {
    setError(null);
    const v = await check();
    if (v?.state !== "download") return;
    const ids = [...new Set([...accepted, ...v.pending.licenses.filter((l) => l.accepted).map((l) => l.id)])];
    if (v.pending.licenses.every((l) => !l.acceptance_required || ids.includes(l.id))) await download(v.pending.token, ids);
  };

  const usable = backend?.info.usable ?? false;
  const downloading = op?.state === "running" && op.kind === "download" && !(op.bytes_total > 0 && op.bytes_done >= op.bytes_total);
  // verifying, activating and testing are one step for the user
  const finishing = !downloading && ((op?.state === "running") || (op?.state === "done" && op.kind !== "probe" && !usable));
  const interrupted = (op?.state === "failed" || op?.state === "cancelled") && op.kind !== "probe" ? op.state : null;

  let body;
  if (downloading && op) {
    const done = gigabytes(op.bytes_done, number);
    const total = gigabytes(op.bytes_total, number);
    body = (
      <>
        {op.bytes_total > 0 ? <progress max={op.bytes_total} value={op.bytes_done} aria-label={t("setup.download_progress")} /> : <progress aria-label={t("setup.download_progress")} />}
        {op.bytes_total > 0 && <p className="muted mono-nums">{eta === null ? t("setup.progress", { done, total }) : t("setup.progress_eta", { done, total, time: etaText(eta, n, t) })}</p>}
        <button onClick={() => api.modelsCancel().catch((e) => setError(errorText(e)))}>{t("setup.pause")}</button>
      </>
    );
  } else if (finishing) {
    body = (
      <>
        <progress aria-label={t("setup.finishing")} />
        <p className="muted" role="status">{t("setup.finishing")}</p>
      </>
    );
  } else if (interrupted) {
    body = (
      <>
        <p className={interrupted === "failed" ? "error" : "muted"} role={interrupted === "failed" ? "alert" : "status"}>{t(interrupted === "failed" ? "setup.failed" : "setup.paused")}</p>
        <button className="primary" onClick={resume}>{t("setup.resume")}</button>
      </>
    );
  } else if (!view) {
    body = slow ? <p className="muted" role="status">{t("setup.checking")}</p> : null;
  } else if (view.state === "download") {
    const p = view.pending;
    const size = gigabytes(p.download_bytes, number);
    const needed = p.licenses.filter((l) => l.acceptance_required && !l.accepted);
    const ok = needed.every((l) => accepted.includes(l.id));
    body = (
      <>
        <p>{t("setup.body", { size })}</p>
        {needed.map((l) => (
          <label key={l.id} className="block">
            <input type="checkbox" checked={accepted.includes(l.id)} onChange={(e) => setAccepted((a) => (e.target.checked ? [...a, l.id] : a.filter((x) => x !== l.id)))} /> {t("setup.agree", { name: `${l.name}${l.spdx ? ` (${l.spdx})` : ""}` })}
            <div className="muted small">{l.url}</div>
          </label>
        ))}
        {!ok && <p className="muted small">{t("setup.accept_hint")}</p>}
        <button className="primary" disabled={!ok} onClick={() => download(p.token)}>{t("setup.download", { size })}</button>
        <p className="muted small">{t("setup.note")}</p>
      </>
    );
  } else if (view.state === "offline") {
    body = (
      <>
        <p role="alert">{hostText(view.reason, t)}</p>
        <button className="primary" onClick={() => void check()}>{t("setup.retry")}</button>
      </>
    );
  } else if (view.state === "unavailable") {
    body = <p className="muted" role="status">{hostText(view.reason, t)}</p>;
  } else if (view.state === "problem") {
    body = (
      <>
        <h3>{t("setup.problem_title")}</h3>
        <p>{reasonText(view.reason, t)}</p>
        <button onClick={onOpenModels}>{t("setup.open_models")}</button>
      </>
    );
  } else {
    body = <p className="muted" role="status">{t("setup.working")}</p>;
  }

  return (
    <section className={first ? "setup welcome-card" : "setup card"}>
      <h2>{view?.state === "unavailable" ? t("setup.unavailable_title") : t("setup.title")}</h2>
      {first && <p className="promise">{t("setup.promise")}</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {body}
    </section>
  );
}
