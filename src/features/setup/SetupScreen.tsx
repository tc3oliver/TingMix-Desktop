import { useCallback, useEffect, useState } from "react";
import { useI18n } from "../../i18n";
import { hostText, reasonText } from "../../i18n/host";
import { api, errorText, onBackend, onModelOp } from "../../lib/api";
import type { BackendState, ModelOp, SetupView } from "../../types/protocol";
import { onboardStage } from "../models/onboard";

interface Props {
  backend: BackendState | null;
  onOpenModels: () => void;
}

/** Size as "1.3 GB" in the UI language (decimal gigabytes, what download sites show). */
export function gigabytes(bytes: number, number: (n: number, o?: Intl.NumberFormatOptions) => string): string {
  return bytes >= 1e9 ? `${number(bytes / 1e9, { maximumFractionDigits: 1 })} GB` : `${number(Math.max(1, bytes / 1e6), { maximumFractionDigits: 0 })} MB`;
}

/**
 * The first-run screen: one download, then everything else happens by itself (signature check, activation and a test on
 * this computer). The host does the work; this screen follows the model operation and backend events.
 */
export function SetupScreen({ backend, onOpenModels }: Props) {
  const { t, number } = useI18n();
  const [view, setView] = useState<SetupView | null>(null);
  const [op, setOp] = useState<ModelOp | null>(null);
  const [accepted, setAccepted] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const check = useCallback(() => {
    setView(null);
    return api.setupCheck().then((v) => {
      setView(v);
      if (v.state === "download") setAccepted(v.pending.licenses.filter((l) => l.accepted).map((l) => l.id));
    }).catch((e) => setError(errorText(e)));
  }, []);

  useEffect(() => {
    void check();
    let alive = true;
    const offs: (() => void)[] = [];
    const keep = (p: Promise<() => void>) => p.then((u) => (alive ? offs.push(u) : u()));
    keep(onModelOp((o) => setOp(o)));
    keep(onBackend(() => void check())); // a finished test changes what is needed
    void api.modelsOverview().then((o) => alive && setOp(o.op));
    return () => {
      alive = false;
      offs.forEach((f) => f());
    };
  }, [check]);

  const running = op?.state === "running";
  const stage = onboardStage(op, backend, t);
  const download = async (token: string) => {
    setError(null);
    try {
      await api.modelsInstall(token, accepted);
    } catch (e) {
      setError(errorText(e));
    }
  };

  let body;
  if (running && stage) {
    body = (
      <>
        {op!.bytes_total > 0 ? <progress max={op!.bytes_total} value={op!.bytes_done} aria-label={stage.text} /> : <progress aria-label={stage.text} />}
        <p className="muted" role="status">{stage.text}</p>
        {op!.kind === "download" && <button onClick={() => api.modelsCancel()}>{t("common.cancel")}</button>}
      </>
    );
  } else if ((op?.state === "failed" || op?.state === "cancelled") && op.kind !== "probe") {
    body = (
      <>
        <p className="error" role="alert">{stage?.text}</p>
        <button className="primary" onClick={check}>{t("setup.resume")}</button>
      </>
    );
  } else if (!view) {
    body = <p className="muted" role="status">{t("setup.checking")}</p>;
  } else if (view.state === "download") {
    const p = view.pending;
    const size = gigabytes(p.download_bytes, number);
    const needed = p.licenses.filter((l) => l.acceptance_required && !l.accepted);
    const ok = needed.every((l) => accepted.includes(l.id));
    body = (
      <>
        <p>{t("setup.body")}</p>
        {needed.map((l) => (
          <label key={l.id} className="block">
            <input type="checkbox" checked={accepted.includes(l.id)} onChange={(e) => setAccepted((a) => (e.target.checked ? [...a, l.id] : a.filter((x) => x !== l.id)))} /> {t("models.accept", { name: `${l.name}${l.spdx ? ` (${l.spdx})` : ""}` })}
            <div className="muted small">{l.url}</div>
          </label>
        ))}
        {!ok && <p className="muted small">{t("setup.accept_hint")}</p>}
        <button className="primary" disabled={!ok} onClick={() => download(p.token)}>{t("setup.download", { size })}</button>
        <p className="muted small">{t("setup.note", { size })}</p>
      </>
    );
  } else if (view.state === "offline") {
    body = (
      <>
        <p className="error" role="alert">{hostText(view.reason, t)}</p>
        <button className="primary" onClick={check}>{t("setup.retry")}</button>
      </>
    );
  } else if (view.state === "unavailable") {
    body = <p role="status">{hostText(view.reason, t)}</p>;
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
    <section className="setup card">
      <h2>{view?.state === "unavailable" ? t("setup.unavailable_title") : t("setup.title")}</h2>
      {error && <p className="error" role="alert">{error}</p>}
      {body}
    </section>
  );
}
