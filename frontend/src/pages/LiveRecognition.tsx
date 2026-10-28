import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, getApiBase } from "../api/client";
import { useApi, useInterval } from "../api/hooks";
import type {
  DatasetsListResponse,
  DatasetSubjectsResponse, ExperimentDetailResponse, ReplayEvent, ReplayStartResponse,
} from "../api/types";
import { useAppState } from "../state/AppState";
import { Badge, Button, Card, EmptyState, ErrorState, Field, SectionNote, Select, StatCard, StatusDot } from "../components/ui";
import { LineChart } from "../components/charts/LineChart";
import { ProbabilityBars } from "../components/charts/ProbabilityBars";
import { RunPicker, modelLabel } from "../components/pickers";
import { fmtMs, fmtPct, prettyLabel } from "../lib/format";

type WsStatus = "idle" | "connecting" | "connected" | "disconnected" | "error";

interface ReplayState {
  session: ReplayStartResponse | null;
  event: ReplayEvent | null;
  history: ReplayEvent[];
  finished: boolean;
}

const MAX_HISTORY = 40;

export default function LiveRecognition() {
  const { experiments, selectedRunId, setSelectedRunId } = useAppState();
  const [runId, setRunId] = useState<string>(selectedRunId ?? "");
  const [dataset, setDataset] = useState<string>("");
  const [subject, setSubject] = useState<string>("");
  const [speed, setSpeed] = useState<number>(1);
  const [maxWindows, setMaxWindows] = useState<number>(80);

  // keep dataset default aligned with the selected run
  useEffect(() => {
    const run = experiments.find((e) => e.run_id === runId);
    if (run) setDataset(run.dataset);
  }, [runId, experiments]);

  const subjectsReq = useApi(
    () => api.get<DatasetSubjectsResponse>(`/datasets/${dataset}/subjects`),
    [dataset],
    { enabled: !!dataset },
  );
  const subjectOptions = useMemo(() => subjectsReq.data?.subjects ?? [], [subjectsReq.data]);

  const [replay, setReplay] = useState<ReplayState>({ session: null, event: null, history: [], finished: false });
  const [paused, setPaused] = useState(false);
  const [wsStatus, setWsStatus] = useState<WsStatus>("idle");
  const [busy, setBusy] = useState<"" | "start" | "play" | "stop" | "step" | "speed">("");
  const [error, setError] = useState<string | null>(null);

  const pausedRef = useRef(paused);
  pausedRef.current = paused;
  const wsRef = useRef<WebSocket | null>(null);
  const stripRef = useRef<{ x: number; y: number }[]>([]);
  const sampleCounter = useRef(0);

  const runDetail = useApi(
    () => api.get<ExperimentDetailResponse>(`/experiments/${runId}`),
    [runId],
    { enabled: !!runId },
  );
  const adapters = useApi(() => api.get<DatasetsListResponse>("/datasets"), []);

  /* ----------------------------- websocket ----------------------------- */
  const handleEvent = useCallback((event: ReplayEvent) => {
    if (pausedRef.current) return; // freeze the display while paused
    const s = event.sensor_sample ?? [];
    for (const v of s) {
      sampleCounter.current += 1;
      stripRef.current.push({ x: sampleCounter.current, y: v });
    }
    if (stripRef.current.length > 128) {
      stripRef.current = stripRef.current.slice(-128);
    }
    setReplay((r) => {
      const top = r.history[0];
      if (top && top.index === event.index && top.timestamp === event.timestamp && r.event) {
        return r; // duplicate from step+stream
      }
      return {
        session: r.session,
        event,
        history: [event, ...r.history].slice(0, MAX_HISTORY),
        finished: false,
      };
    });
  }, []);

  const openWs = useCallback((sessionId: string) => {
    if (wsRef.current) {
      try { wsRef.current.close(); } catch { /* ignore */ }
    }
    const base = getApiBase().replace(/^http/, "ws");
    let ws: WebSocket;
    try {
      ws = new WebSocket(`${base}/ws/replay/${sessionId}`);
    } catch {
      setWsStatus("disconnected");
      return;
    }
    setWsStatus("connecting");
    wsRef.current = ws;
    ws.onopen = () => {
      setWsStatus("connected");
      ws.send("open");
    };
    ws.onmessage = (e) => {
      let msg: Record<string, unknown>;
      try { msg = JSON.parse(String(e.data)); } catch { return; }
      if (msg.event === "heartbeat") {
        ws.send("ack");
        return;
      }
      if (msg.event === "end") {
        setWsStatus("disconnected");
        setReplay((r) => ({ ...r, finished: true }));
        return;
      }
      handleEvent(msg as unknown as ReplayEvent);
      ws.send("ack");
    };
    ws.onerror = () => setWsStatus("disconnected");
    ws.onclose = () => setWsStatus((s) => (s === "connecting" ? "disconnected" : "disconnected"));
  }, [handleEvent]);

  // keep the stream flowing: the server loop waits for one client message per
  // cycle, so a steady ping stream drives it.
  useInterval(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && !pausedRef.current) {
      wsRef.current.send("ping");
    }
  }, 400);

  useEffect(() => {
    return () => {
      try { wsRef.current?.close(); } catch { /* ignore */ }
    };
  }, []);

  /* ----------------------------- controls ----------------------------- */
  async function startSession() {
    setError(null);
    setBusy("start");
    try {
      if (replay.session) {
        await api.post("/replay/stop", { session_id: replay.session.session_id }, 10000).catch(() => undefined);
      }
      const payload: Record<string, unknown> = { speed };
      if (runId) payload.run_id = runId;
      if (dataset) payload.dataset = dataset;
      if (subject) payload.subject = subject;
      if (maxWindows > 0) payload.max_windows = maxWindows;
      const s = await api.post<ReplayStartResponse>("/replay/start", payload, 180000);
      stripRef.current = [];
      sampleCounter.current = 0;
      setReplay({ session: s, event: null, history: [], finished: false });
      setPaused(false);
      if (runId) {
        setSelectedRunId(runId);
        // autoplay immediately after start (predictor warm-up done server-side)
        await api.post("/replay/play", { session_id: s.session_id, speed }, 10000).catch(() => undefined);
        openWs(s.session_id);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  }

  async function togglePlay() {
    if (!replay.session) return;
    setError(null);
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      // playing → pause (freeze local rendering; backend keeps stepping)
      setPaused(true);
    } else {
      setBusy("play");
      try {
        await api.post("/replay/play", { session_id: replay.session.session_id, speed }, 10000);
        setPaused(false);
        openWs(replay.session.session_id);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setBusy("");
      }
    }
  }

  async function stopSession() {
    if (!replay.session) return;
    setBusy("stop");
    setError(null);
    try {
      await api.post("/replay/stop", { session_id: replay.session.session_id }, 10000);
      try { wsRef.current?.close(); } catch { /* ignore */ }
      wsRef.current = null;
      setWsStatus("idle");
      setReplay({ session: null, event: null, history: [], finished: false });
      setPaused(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  }

  async function stepOnce() {
    if (!replay.session) return;
    setBusy("step");
    setError(null);
    try {
      const ev = await api.post<ReplayEvent>("/replay/step", { session_id: replay.session.session_id }, 120000);
      handleEvent(ev);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  }

  async function changeSpeed(v: number) {
    setSpeed(v);
    if (!replay.session) return;
    setBusy("speed");
    try {
      await api.post("/replay/control", { session_id: replay.session.session_id, speed: v }, 10000);
    } catch {
      /* non-critical */
    } finally {
      setBusy("");
    }
  }

  /* ----------------------------- derived ----------------------------- */
  const ev = replay.event;
  const prediction = ev?.prediction ?? null;
  const hasModel = !!runId;
  const labeledCount = replay.history.filter((h) => h.correct !== null).length;
  const correctCount = replay.history.filter((h) => h.correct === true).length;
  const isPlaying = wsRef.current !== null && (wsStatus === "connected" || wsStatus === "connecting") && !paused;
  const canPause = isPlaying;
  const strip = stripRef.current;
  const curSubject = ev?.subject_id;
  const idx = ev ? ev.index : 0;
  const nWin = replay.session?.n_windows ?? 0;
  const conf = prediction?.confidence ?? null;
  const angle = conf != null ? Math.min(0.92, Math.max(0.04, conf)) * 180 : null;

  const gauge = (angle: number | null) => {
    const r = 46, cx = 56, cy = 52;
    const a = angle == null ? 0 : (angle / 180) * Math.PI;
    const x = cx - r * Math.cos(a);
    const y = cy - r * Math.sin(a);
    const large = angle != null && angle > 157 ? 1 : 0;
    return angle == null ? null : `M${cx} ${cy} L${cx - r} ${cy} A${r} ${r} 0 ${large} 1 ${x.toFixed(2)} ${y.toFixed(2)}`;
  };

  const running = !!replay.session && !replay.finished;

  return (
    <>
      <div className="page-head">
        <div>
          <div className="page-title">Live Recognition</div>
          <div className="page-desc">
            Replay dataset windows through a trained model. The backend simulates a live sensor
            feed; predictions stream over WebSocket. Confidence is the softmax probability of the
            predicted class — never call it accuracy.
          </div>
        </div>
        <div className="row">
          <Badge tone={wsStatus === "connected" ? "green" : wsStatus === "connecting" ? "amber" : ""}>
            <StatusDot status={wsStatus === "connected" ? "ok" : wsStatus === "disconnected" ? "err" : "warn"} pulse={wsStatus === "connected"} />
            WS {wsStatus}
          </Badge>
          {running && <Badge tone="cyan">session {replay.session?.session_id}</Badge>}
        </div>
      </div>

      {/* controls */}
      <Card title="Replay controls" sub="Attach a trained run to get predictions, or replay signals only">
        <div className="toolbar" style={{ marginBottom: 10 }}>
          <RunPicker
            experiments={experiments}
            value={runId || null}
            onChange={setRunId}
            label="Model run"
            style={{ minWidth: 260 }}
          />
          <Field label="Dataset" style={{ minWidth: 150 }}>
            <Select
              value={dataset}
              placeholder={runId ? "(from run)" : "select dataset"}
              onChange={(v) => { setDataset(v); setSubject(""); }}
              options={(adapters.data?.adapters ?? []).map((d) => ({ value: d, label: d }))}
              disabled={!!runId}
            />
          </Field>
          <Field label="Subject filter" style={{ minWidth: 150 }}>
            <Select
              value={subject}
              placeholder="All subjects"
              onChange={setSubject}
              options={subjectOptions.map((s) => ({ value: s, label: s }))}
            />
          </Field>
          <Field label="Max windows" style={{ width: 110 }}>
            <input
              className="input mono" type="number" min={0} max={5000}
              value={maxWindows}
              onChange={(e) => setMaxWindows(Math.max(0, Math.floor(Number(e.target.value) || 0)))}
            />
          </Field>
          <Field label={`Speed ×${speed.toFixed(2)}`} style={{ width: 170 }}>
            <input
              type="range" min={0.25} max={5} step={0.25} value={speed}
              onChange={(e) => void changeSpeed(Number(e.target.value))}
              style={{ width: "100%", accentColor: "var(--accent)" }}
            />
          </Field>
          <div className="row" style={{ paddingBottom: 2 }}>
            <Button variant="primary" disabled={busy !== "" || (!runId && !dataset)} onClick={() => void startSession()}>
              {busy === "start" ? "Starting…" : "Start replay"}
            </Button>
            <Button disabled={!running || busy !== ""} onClick={() => void stepOnce()} title="Advance one window (uses /replay/step)">
              {busy === "step" ? "…" : "Step"}
            </Button>
            <Button variant={canPause ? "" : "primary"} disabled={!running || busy !== "" || (!canPause && wsStatus !== "disconnected")} onClick={() => void togglePlay()}>
              {canPause ? "Pause stream" : "Resume stream"}
            </Button>
            <Button variant="danger" disabled={!running || busy !== ""} onClick={() => void stopSession()}>
              {busy === "stop" ? "…" : "Stop"}
            </Button>
          </div>
        </div>
        {error && <div className="error-banner"><span className="code">REPLAY</span><span>{error}</span></div>}
        {paused && (
          <div style={{ marginTop: 8 }}>
            <SectionNote tone="amber">
              <span><b>Stream paused.</b> The display is frozen; the backend keeps stepping ahead.
              Resume re-syncs to the latest window.</span>
            </SectionNote>
          </div>
        )}
        {!running && !replay.finished && (
          <div style={{ marginTop: 4 }}>
            <SectionNote>
              <span><b>How it works:</b> press <i>Start replay</i> — the backend builds a window
              source from the selected run's configuration and begins streaming predictions. Use
              <i> Step</i> for single-window analysis during a viva.</span>
            </SectionNote>
          </div>
        )}
      </Card>

      {replay.finished && (
        <div style={{ marginTop: 12 }}>
          <Card>
            <EmptyState
              message="Replay finished"
              hint="The source ran out of windows. Start again or pick another subject."
              icon={<span style={{ fontSize: 26 }}>✔</span>}
              actions={<Button size="sm" onClick={() => void startSession()}>Restart replay</Button>}
            />
          </Card>
        </div>
      )}

      {/* main display */}
      <div className="grid grid-c3" style={{ marginTop: 14 }}>
        {/* hero */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card title="Current activity" sub={ev ? `window ${ev.window_id} · index ${idx}${nWin ? `/${nWin - 1}` : ""}` : "waiting for replay"}>
            <div className="hero-activity">
              <span className="hero-label">predicted activity</span>
              <div className="hero-value">
                {prediction ? prediction.predicted_activity.replace(/_/g, " ") : hasModel ? "…" : <small>no model attached</small>}
              </div>
              {ev?.true_activity && (
                <span className="hint">
                  ground truth: <b style={{ color: "var(--text-dim)" }}>{prettyLabel(ev.true_activity)}</b>
                </span>
              )}
              {ev?.correct != null ? (
                <Badge tone={ev.correct ? "green" : "red"}>{ev.correct ? "✓ correct" : "✗ incorrect"}</Badge>
              ) : ev ? (
                <Badge>{hasModel ? "unlabeled — no truth" : "signal replay only"}</Badge>
              ) : null}
            </div>
          </Card>

          <Card title="Confidence" sub="softmax probability of the predicted class">
            <div className="gauge-wrap">
              <svg width="112" height="76" viewBox="0 0 112 76">
                <path d={`M10 52 A${46} ${46} 0 0 1 102 52`} fill="none" stroke="var(--panel-3)" strokeWidth={9} strokeLinecap="round" />
                {gauge(angle) && (
                  <path
                    d={gauge(angle)!}
                    fill="none"
                    stroke={conf != null && conf > 0.75 ? "var(--green)" : conf != null && conf > 0.5 ? "var(--amber)" : "var(--red)"}
                    strokeWidth={9}
                    strokeLinecap="round"
                    style={{ transition: "d 0.3s ease" }}
                  />
                )}
                <text x={56} y={62} textAnchor="middle" fontSize={16} fontWeight={700} fontFamily="var(--mono)" fill="var(--text)">
                  {conf != null ? fmtPct(conf, 1) : "—"}
                </text>
              </svg>
            </div>
            {prediction && (
              <div className="kv" style={{ gridTemplateColumns: "auto auto", gap: "3px 12px", marginTop: 4 }}>
                <dt>TDA branch</dt><dd>{fmtMs(prediction.inference_time_ms > 0 ? (prediction as unknown as { tda_ms: number }).tda_ms : 0)}</dd>
                <dt>Deep branch</dt><dd>{fmtMs(prediction.inference_time_ms > 0 ? (prediction as unknown as { deep_ms: number }).deep_ms : 0)}</dd>
                <dt>Fusion head</dt><dd>{fmtMs(prediction.inference_time_ms > 0 ? (prediction as unknown as { fusion_ms: number }).fusion_ms : 0)}</dd>
                <dt>Total</dt><dd style={{ color: "var(--accent)" }}>{fmtMs(prediction.inference_time_ms)}</dd>
              </div>
            )}
          </Card>
        </div>

        {/* probabilities */}
        <Card title="Class probabilities" sub={ev ? `model output for window ${ev.window_id}` : "waiting for predictions"}>
          {prediction ? (
            <ProbabilityBars probabilities={prediction.probabilities} highlight={prediction.predicted_activity} />
          ) : (
            <EmptyState
              message="No prediction yet"
              hint={hasModel ? "Start the replay to stream predictions." : "Attach a trained run to enable predictions."}
              icon={<span style={{ fontSize: 24 }}>%</span>}
            />
          )}
          {prediction && (
            <div style={{ marginTop: 12 }}>
              <div className="hint" style={{ marginBottom: 6 }}>
                Per-window latency profile (ms) — from the backend's branch-level timers
              </div>
              <div style={{ display: "flex", height: 14, borderRadius: 7, overflow: "hidden", border: "1px solid var(--border)" }}>
                {(() => {
                  const tda = (prediction as unknown as { tda_ms?: number }).tda_ms ?? 0;
                  const deep = (prediction as unknown as { deep_ms?: number }).deep_ms ?? 0;
                  const fus = (prediction as unknown as { fusion_ms?: number }).fusion_ms ?? 0;
                  const tot = Math.max(1e-9, tda + deep + fus);
                  return (
                    <>
                      <div style={{ width: `${(tda / tot) * 100}%`, background: "var(--violet)", opacity: 0.8 }} title={`TDA ${fmtMs(tda)}`} />
                      <div style={{ width: `${(deep / tot) * 100}%`, background: "var(--amber)", opacity: 0.8 }} title={`Deep ${fmtMs(deep)}`} />
                      <div style={{ width: `${(fus / tot) * 100}%`, background: "var(--pink)", opacity: 0.8 }} title={`Fusion ${fmtMs(fus)}`} />
                    </>
                  );
                })()}
              </div>
              <div className="legend" style={{ marginTop: 6 }}>
                <span className="legend-item on"><span className="swatch" style={{ background: "var(--violet)" }} />TDA</span>
                <span className="legend-item on"><span className="swatch" style={{ background: "var(--amber)" }} />DeepConvLSTM</span>
                <span className="legend-item on"><span className="swatch" style={{ background: "var(--pink)" }} />Fusion</span>
              </div>
            </div>
          )}
        </Card>

        {/* live signal */}
        <Card
          title="Live signal"
          sub={ev ? `window shape (${ev.window_shape[0]} × ${ev.window_shape[1]}) — rolling strip of the last 8 samples of channel 0 per window` : "waiting for replay"}
        >
          {strip.length > 1 ? (
            <LineChart
              series={[{ name: "channel 0", color: "var(--cyan)", points: strip }]}
              height={210}
              xLabel="sample"
              yLabel="value"
              area
            />
          ) : (
            <EmptyState message="No samples yet" hint="Samples appear once the replay stream delivers windows." icon={<span style={{ fontSize: 22 }}>∿</span>} />
          )}
        </Card>
      </div>

      {/* session stats + history */}
      <div className="grid grid-c2" style={{ marginTop: 0 }}>
        <Card title="Session summary" sub="Aggregated over labeled replay windows only">
          <div className="grid grid-c3" style={{ gap: 10 }}>
            <StatCard label="Windows streamed" value={replay.history.length} accent="var(--blue)" />
            <StatCard
              label="Labeled accuracy (session)"
              accent="var(--green)"
              value={labeledCount ? fmtPct(correctCount / labeledCount) : "—"}
              sub={labeledCount ? `${correctCount}/${labeledCount} labeled windows` : "no labeled windows yet"}
            />
            <StatCard
              label="Avg latency"
              accent="var(--amber)"
              value={replay.history.some((h) => h.prediction) ? fmtMs(replay.history.filter((h) => h.prediction).reduce((a, h) => a + h.prediction!.inference_time_ms, 0) / Math.max(1, replay.history.filter((h) => h.prediction).length)) : "—"}
              sub="mean inference ms this session"
            />
          </div>
          <div style={{ marginTop: 10 }}>
            <SectionNote>
              <span><b>Honest reporting:</b> “Labeled accuracy (session)” is computed only over
              windows that carry ground truth and is a session statistic, not a model evaluation.
              Formal metrics live in the <b>Evaluation</b> page.</span>
            </SectionNote>
          </div>
        </Card>

        <Card title="Prediction history" sub="Most recent first">
          {replay.history.length === 0 ? (
            <EmptyState message="Empty stream" hint="Predictions will be listed here as they stream in." />
          ) : (
            <div style={{ maxHeight: 300, overflowY: "auto" }}>
              <table className="table">
                <thead>
                  <tr><th>Idx</th><th>Subject</th><th>Truth</th><th>Predicted</th><th className="num">Conf</th><th className="num">ms</th><th /></tr>
                </thead>
                <tbody>
                  {replay.history.map((h) => (
                    <tr key={`${h.index}-${h.timestamp}`}>
                      <td>{h.index}</td>
                      <td>{h.subject_id}</td>
                      <td>{h.true_activity ? prettyLabel(h.true_activity) : "—"}</td>
                      <td style={{ color: "var(--text)" }}>{prettyLabel(h.prediction?.predicted_activity ?? "—")}</td>
                      <td className="num">{h.prediction ? fmtPct(h.prediction.confidence) : "—"}</td>
                      <td className="num">{h.prediction ? fmtMs(h.prediction.inference_time_ms) : "—"}</td>
                      <td>{h.correct == null ? "" : h.correct ? "✓" : "✗"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      {/* run info */}
      {runDetail.data && (
        <Card title={`Replaying with ${modelLabel(runDetail.data.experiment.model_name)} — ${runDetail.data.run_id}`} style={{ marginTop: 14 }}>
          <div className="hint">
            window {fmtMs((runDetail.data.experiment.window_duration ?? 0) * 1000, 0)} · {runDetail.data.experiment.window_size} samples ·
            step {runDetail.data.experiment.step_size} · {runDetail.data.experiment.sampling_rate} Hz ·
            TDA {runDetail.data.experiment.tda_enabled ? "enabled" : "disabled"} ·
            eval fold {runDetail.data.summary?.fold ?? "N/A"}
          </div>
        </Card>
      )}
      {runDetail.error && <div style={{ marginTop: 14 }}><ErrorState error={runDetail.error} context="Run details" retry={runDetail.reload} /></div>}
    </>
  );
}
