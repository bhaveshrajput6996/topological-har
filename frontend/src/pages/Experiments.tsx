import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { useApi, useInterval } from "../api/hooks";
import type {
  DatasetsListResponse, ExperimentDetailResponse, ExperimentRow, TrainStartResponse, TrainStatusResponse,
} from "../api/types";
import { useAppState } from "../state/AppState";
import { Badge, Button, Card, ErrorState, Field, SectionNote, Select, StatCard, StatusDot } from "../components/ui";
import { RunPicker, modelLabel } from "../components/pickers";
import { fmtMb, fmtMs, fmtPct, fmtS, fmtTime, metricOrNull } from "../lib/format";
import { navigate } from "../state/router";

type TrainingJob = {
  experimentId: string;
  state: string;
  error?: string;
  nRuns?: number;
};

export default function Experiments() {
  const { experiments, experimentsLoading, refreshExperiments, selectedRunId, setSelectedRunId } = useAppState();
  const [sortKey, setSortKey] = useState<"created" | "accuracy" | "f1">("created");
  const [detailRun, setDetailRun] = useState<string | null>(selectedRunId);
  const detail = useApi(
    () => api.get<ExperimentDetailResponse>(`/experiments/${detailRun}`),
    [detailRun],
    { enabled: !!detailRun },
  );

  const sorted = [...experiments].sort((a, b) => {
    if (sortKey === "created") return b.created_at.localeCompare(a.created_at);
    if (sortKey === "accuracy") return (metricOrNull(b.accuracy) ?? -1) - (metricOrNull(a.accuracy) ?? -1);
    return (metricOrNull(b.f1_macro) ?? -1) - (metricOrNull(a.f1_macro) ?? -1);
  });

  return (
    <>
      <div className="page-head">
        <div>
          <div className="page-title">Experiments</div>
          <div className="page-desc">
            Persistent experiment history backed by the backend's SQLite store
            (<span className="mono">artifacts/har_experiments.db</span>). Runs are never overwritten or
            deleted — every training run appends a row.
          </div>
        </div>
        <div className="row">
          <Button size="sm" onClick={() => void refreshExperiments()}>{experimentsLoading ? "…" : "Refresh"}</Button>
        </div>
      </div>

      <div className="grid grid-c2" style={{ alignItems: "start" }}>
        {/* launch training */}
        <TrainingLauncher onLaunched={() => void refreshExperiments()} />

        {/* history table */}
        <Card
          title={`Experiment history (${experiments.length})`}
          sub="Click a row to inspect; metrics missing from the store display as N/A"
          tools={
            <Select
              value={sortKey}
              onChange={(v) => setSortKey(v as typeof sortKey)}
              options={[
                { value: "created", label: "Sort: newest" },
                { value: "accuracy", label: "Sort: accuracy" },
                { value: "f1", label: "Sort: F1 macro" },
              ]}
              style={{ width: 160, paddingTop: 5, paddingBottom: 5 }}
            />
          }
        >
          {experimentsLoading ? (
            <div className="skeleton" style={{ height: 200 }} />
          ) : experiments.length === 0 ? (
            <div className="hint" style={{ padding: 12 }}>
              No experiments recorded yet. Launch a training run or use the backend CLI.
            </div>
          ) : (
            <div style={{ maxHeight: 420, overflowY: "auto" }} className="scroll-x">
              <table className="table">
                <thead>
                  <tr>
                    <th>Run ID</th><th>Model</th><th>Dataset</th><th className="num">Fold</th>
                    <th className="num">Acc</th><th className="num">F1</th><th className="num">ms</th><th>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((e) => (
                    <tr
                      key={e.run_id}
                      className={`clickable ${detailRun === e.run_id ? "selected" : ""}`}
                      onClick={() => setDetailRun(e.run_id)}
                    >
                      <td>{e.run_id}</td>
                      <td>{modelLabel(e.model)}</td>
                      <td>{e.dataset}</td>
                      <td className="num">{e.fold >= 0 ? e.fold : "—"}</td>
                      <td className="num">{fmtPct(metricOrNull(e.accuracy))}</td>
                      <td className="num">{fmtPct(metricOrNull(e.f1_macro))}</td>
                      <td className="num">{e.avg_inference_ms > 0 ? fmtMs(e.avg_inference_ms) : "—"}</td>
                      <td>{fmtTime(e.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      {/* detail */}
      {detailRun && (
        <div style={{ marginTop: 14 }}>
          {detail.loading ? (
            <Card title={`Run ${detailRun}`}><div className="skeleton" style={{ height: 140 }} /></Card>
          ) : detail.error ? (
            <Card title={`Run ${detailRun}`}><ErrorState error={detail.error} context="Experiment detail" retry={detail.reload} /></Card>
          ) : detail.data ? (
            <ExperimentDetailCard
              data={detail.data}
              onUseRun={() => { setSelectedRunId(detail.data!.run_id); setDetailRun(detail.data!.run_id); }}
              onEvaluate={() => { setSelectedRunId(detail.data!.run_id); navigate("evaluation", { run: detail.data!.run_id }); }}
            />
          ) : null}
        </div>
      )}
    </>
  );
}

/* ------------------------------------------------------------------ */
function ExperimentDetailCard({
  data, onUseRun, onEvaluate,
}: {
  data: ExperimentDetailResponse;
  onUseRun: () => void;
  onEvaluate: () => void;
}) {
  const e = data.experiment;
  const s = data.summary;
  const split = data.split;
  const cfg = (e.config_json ?? {}) as Record<string, any>;
  return (
    <Card
      title={`${e.run_id} — ${e.experiment_name}`}
      sub={`created ${fmtTime(e.created_at)}`}
      tools={
        <div className="row">
          <Button size="sm" variant="ghost" onClick={onUseRun}>Set as active run</Button>
          <Button size="sm" variant="ghost" onClick={onEvaluate}>Open evaluation →</Button>
        </div>
      }
    >
      <div className="grid grid-c4" style={{ gap: 10 }}>
        <StatCard label="Model" value={modelLabel(e.model_name)} accent="var(--amber)" />
        <StatCard label="Dataset" value={e.dataset} accent="var(--blue)" />
        <StatCard label="Window" value={`${fmtS(e.window_duration)} / ${e.window_size} samples`} sub={`step ${e.step_size} · ${e.sampling_rate} Hz`} />
        <StatCard label="TDA" value={e.tda_enabled ? `${e.tda_points} pts · ${e.tda_resolution}` : "disabled"} accent="var(--violet)" />
      </div>
      <div className="divider" />
      <div className="grid grid-c3" style={{ gap: 10 }}>
        <StatCard label="Accuracy" value={fmtPct(metricOrNull(s?.accuracy))} sm />
        <StatCard label="F1 macro / weighted" value={`${fmtPct(metricOrNull(s?.f1_macro))} / ${fmtPct(metricOrNull(s?.f1_weighted))}`} sm />
        <StatCard label="Latency / RAM" value={`${s?.avg_inference_ms && s.avg_inference_ms > 0 ? fmtMs(s.avg_inference_ms) : "N/A"} · ${fmtMb(metricOrNull(s?.peak_ram_mb))}`} sm />
      </div>
      {split.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <div className="hint" style={{ marginBottom: 6 }}>Split metadata (subject-aware)</div>
          <div className="chip-list">
            {split.map((sp) => (
              <span key={sp.fold} className={`chip ${sp.leakage_passed ? "acc" : ""}`} title={`leakage check: ${sp.leakage_passed ? "passed" : "failed"}`}>
                fold {sp.fold}: {sp.train_subjects.length} train / {sp.val_subjects.length} val / {sp.test_subjects.length} test subjects
              </span>
            ))}
          </div>
        </div>
      )}
      {cfg.project?.name && (
        <div style={{ marginTop: 10 }} className="hint">
          config project: <span className="mono">{String(cfg.project.name)}</span> · seed {String(cfg.project.seed ?? 42)}
        </div>
      )}
    </Card>
  );
}

/* ------------------------------------------------------------------ */
function TrainingLauncher({ onLaunched }: { onLaunched: () => void }) {
  const adapters = useApi(() => api.get<DatasetsListResponse>("/datasets"), []);
  const [name, setName] = useState("dashboard_run");
  const [dataset, setDataset] = useState("synthetic");
  const [model, setModel] = useState("tda_deepconvlstm");
  const [duration, setDuration] = useState("2.0");
  const [epochs, setEpochs] = useState("30");
  const [folds, setFolds] = useState("0");
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const pollRef = useRef<number | null>(null);

  useInterval(
    () => {
      if (!job || (job.state !== "running" && job.state !== "submitted")) return;
      void (async () => {
        try {
          const st = await api.get<TrainStatusResponse>(`/experiments/status/${job.experimentId}`, 10000);
          setJob((j) => (j && j.experimentId === job.experimentId ? { ...j, state: st.state, error: st.error, nRuns: st.runs?.length } : j));
          if (st.state === "done" || st.state === "failed") onLaunched();
        } catch {
          /* transient */
        }
      })();
    },
    job && (job.state === "running" || job.state === "submitted") ? 2500 : null,
  );

  useEffect(() => () => { if (pollRef.current) window.clearInterval(pollRef.current); }, []);

  const isSvm = model === "svm";
  const isUci = dataset === "uci_har";

  async function launch() {
    setStarting(true);
    setError(null);
    try {
      const config: Record<string, unknown> = {
        project: { name: name || "dashboard_run", seed: 42 },
        dataset: { name: dataset, signal_mode: isSvm && isUci ? "engineered" : "inertial" },
        window: { duration_seconds: Number(duration) || 2.0, overlap: null },
        model: { type: model },
        training: { epochs: Math.max(1, Number(epochs) || 30), patience: 6 },
        evaluation: { cv_folds: 5, folds: folds === "" ? null : folds.split(",").map((f) => Number(f.trim())).filter((n) => !Number.isNaN(n)) },
      };
      const res = await api.post<TrainStartResponse>("/experiments", { config, name: name || undefined }, 20000);
      setJob({ experimentId: res.experiment_id, state: res.state });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setStarting(false);
    }
  }

  const running = job && (job.state === "running" || job.state === "submitted");

  return (
    <Card title="Launch training run" sub="Builds a valid inline config and POSTs it to /experiments (background worker)">
      <div className="row" style={{ alignItems: "flex-end", gap: 10 }}>
        <Field label="Name" style={{ minWidth: 130 }}>
          <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
        </Field>
        <Field label="Dataset" style={{ minWidth: 120 }}>
          <Select value={dataset} onChange={setDataset} options={(adapters.data?.adapters ?? []).map((d) => ({ value: d, label: d }))} />
        </Field>
        <Field label="Model" style={{ minWidth: 170 }}>
          <Select
            value={model}
            onChange={setModel}
            options={[
              { value: "tda_deepconvlstm", label: "TDA + DeepConvLSTM" },
              { value: "deepconvlstm", label: "DeepConvLSTM" },
              { value: "svm", label: "SVM (engineered)" },
              { value: "cnn", label: "CNN" },
              { value: "lstm", label: "LSTM" },
            ]}
          />
        </Field>
        <Field label="Window (s)" style={{ width: 90 }}>
          <input className="input mono" type="number" step={0.1} min={0.2} value={duration} onChange={(e) => setDuration(e.target.value)} />
        </Field>
        <Field label="Epochs" style={{ width: 80 }}>
          <input className="input mono" type="number" min={1} max={200} value={epochs} onChange={(e) => setEpochs(e.target.value)} disabled={isSvm} />
        </Field>
        <Field label="Folds (0=all)" style={{ width: 100 }}>
          <input className="input mono" value={folds} onChange={(e) => setFolds(e.target.value)} />
        </Field>
        <Button variant="primary" disabled={starting || !!running} onClick={() => void launch()}>
          {starting ? "Submitting…" : running ? "Running…" : "Start training"}
        </Button>
      </div>
      {isUci && !isSvm && (
        <div style={{ marginTop: 10 }}>
          <SectionNote>
            <span><b>UCI note:</b> windows are dataset-presegmented (2.56 s @ 50 Hz with 50%
            overlap). Shorter durations (e.g. 1.5 s) crop each presegmented row; longer ones are
            refused honestly by the backend.</span>
          </SectionNote>
        </div>
      )}
      {error && <div style={{ marginTop: 10 }}><div className="error-banner"><span className="code">TRAIN</span><span>{error}</span></div></div>}
      {job && (
        <div className="row" style={{ marginTop: 12 }}>
          <Badge tone={job.state === "done" ? "green" : job.state === "failed" ? "red" : "amber"}>
            <StatusDot status={job.state === "done" ? "ok" : job.state === "failed" ? "err" : "warn"} pulse={!!running} />
            {job.experimentId}: {job.state}
          </Badge>
          {running && <span className="hint">polling /experiments/status every 2.5 s — training runs in the backend process</span>}
          {job.state === "failed" && job.error && (
            <div className="hint" style={{ color: "var(--red)" }}>{job.error}</div>
          )}
          {job.state === "done" && (
            <span className="hint">
              finished — {job.nRuns ?? 0} fold run(s) recorded. <a onClick={() => { onLaunched(); }}>Refresh list</a>
            </span>
          )}
        </div>
      )}
    </Card>
  );
}
