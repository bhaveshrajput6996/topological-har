import { api } from "../api/client";
import { useApi } from "../api/hooks";
import type {
  ConfusionResponse, ExperimentDetailResponse, MetricsResponse, TrainingHistoryResponse, WindowHistoryResponse,
} from "../api/types";
import { useAppState } from "../state/AppState";
import { Badge, Button, Card, EmptyState, ErrorState, SectionNote, StatCard, Skeleton } from "../components/ui";
import { ConfusionMatrix } from "../components/charts/ConfusionMatrix";
import { LineChart } from "../components/charts/LineChart";
import { RunPicker, modelLabel } from "../components/pickers";
import { fmtMb, fmtMs, fmtNum, fmtPct, fmtS, fmtTime, metricOrNull, prettyLabel } from "../lib/format";

export default function Evaluation() {
  const { experiments, selectedRunId, setSelectedRunId, selectedRun } = useAppState();
  const runId = selectedRunId;

  const detail = useApi(
    () => api.get<ExperimentDetailResponse>(`/experiments/${runId}`),
    [runId],
    { enabled: !!runId },
  );
  const metrics = useApi(
    () => api.get<MetricsResponse>(`/experiments/${runId}/metrics`),
    [runId],
    { enabled: !!runId },
  );
  const confusion = useApi(
    () => api.get<ConfusionResponse>(`/experiments/${runId}/confusion-matrix`),
    [runId],
    { enabled: !!runId },
  );
  const training = useApi(
    () => api.get<TrainingHistoryResponse>(`/experiments/${runId}/training-history`),
    [runId],
    { enabled: !!runId },
  );
  const history = useApi(
    () => api.get<WindowHistoryResponse>(`/experiments/${runId}/history?limit=400`),
    [runId],
    { enabled: !!runId },
  );

  if (!runId) {
    return (
      <Card title="Evaluation">
        <EmptyState
          message="No run selected"
          hint="Choose an experiment run to see its stored evaluation metrics."
          icon={<span style={{ fontSize: 24 }}>📊</span>}
          actions={<RunPicker experiments={experiments} value={null} onChange={setSelectedRunId} label="" />}
        />
      </Card>
    );
  }

  const s = detail.data?.summary ?? metrics.data?.summary ?? null;
  const m = metrics.data?.metrics?.metrics ?? null;
  const perClass = m?.per_class ?? {};

  const latency = m?.inference_latency_ms ?? null;
  const hasAnyMetrics = s != null || m != null;

  return (
    <>
      <div className="page-head">
        <div>
          <div className="page-title">Evaluation</div>
          <div className="page-desc">
            Stored evaluation of run <span className="mono">{runId}</span> — aggregate test metrics,
            per-class breakdown, confusion matrix, training curves and resource profile. Values come
            from the backend's SQLite store and metrics artifacts; missing values show N/A.
          </div>
        </div>
        <RunPicker
          experiments={experiments}
          value={runId}
          onChange={setSelectedRunId}
          label="Run"
          style={{ minWidth: 280 }}
        />
      </div>

      {detail.error && <Card title="Run"><ErrorState error={detail.error} context="Run detail" retry={detail.reload} /></Card>}

      {/* headline metrics */}
      <div className="grid grid-c4">
        <StatCard label="Test accuracy" accent="var(--green)" value={s ? fmtPct(metricOrNull(s.accuracy)) : "N/A"} sub={s?.fold != null && s.fold >= 0 ? `fold ${s.fold}` : "aggregate"} />
        <StatCard label="Precision (macro)" accent="var(--cyan)" value={s ? fmtPct(metricOrNull(s.precision_macro)) : "N/A"} sub={s ? `weighted ${fmtPct(metricOrNull(s.precision_weighted))}` : undefined} />
        <StatCard label="Recall (macro)" accent="var(--amber)" value={s ? fmtPct(metricOrNull(s.recall_macro)) : "N/A"} sub={s ? `weighted ${fmtPct(metricOrNull(s.recall_weighted))}` : undefined} />
        <StatCard label="F1 (macro)" accent="var(--violet)" value={s ? fmtPct(metricOrNull(s.f1_macro)) : "N/A"} sub={s ? `weighted ${fmtPct(metricOrNull(s.f1_weighted))}` : undefined} />
      </div>

      <div className="grid grid-c4" style={{ marginTop: 0 }}>
        <StatCard label="Avg inference latency" accent="var(--blue)" value={s?.avg_inference_ms != null && s.avg_inference_ms > 0 ? fmtMs(s.avg_inference_ms) : "N/A"} sub={latency ? `p95 ${fmtMs(latency.p95_ms)} · max ${fmtMs(latency.max_ms)}` : "per-window evaluation"} />
        <StatCard label="Training time" accent="var(--amber)" value={s ? fmtS(metricOrNull(s.training_time_s)) : "N/A"} sub={training.data?.epochs?.length ? `${training.data.epochs.length} epochs logged` : undefined} />
        <StatCard label="Peak RAM" accent="var(--pink)" value={s ? fmtMb(metricOrNull(s.peak_ram_mb)) : "N/A"} sub="run resource profile" />
        <StatCard label="Windows evaluated" accent="var(--cyan)" value={s?.num_windows ?? "N/A"} sub={history.data ? `${history.data.count} in window log` : undefined} />
      </div>

      {!hasAnyMetrics && !metrics.loading && (
        <div style={{ marginTop: 14 }}>
          <Card>
            <EmptyState
              message="Data unavailable"
              hint="This run has no stored evaluation summary. It may still be training or the metrics artifact is missing."
              icon={<span style={{ fontSize: 22 }}>∅</span>}
              actions={<Button size="sm" onClick={() => { metrics.reload(); detail.reload(); }}>Re-check</Button>}
            />
          </Card>
        </div>
      )}

      <div className="grid grid-c2" style={{ marginTop: 14 }}>
        {/* confusion matrix */}
        <Card title="Confusion matrix" sub="Rows: true activity · columns: predicted activity · coloring = row-normalized share">
          {confusion.loading ? (
            <Skeleton h={240} />
          ) : confusion.error ? (
            <ErrorState error={confusion.error} context="Confusion matrix" retry={confusion.reload} />
          ) : (confusion.data?.confusion_matrix?.length ?? 0) === 0 ? (
            <EmptyState message="Data unavailable" hint="No confusion matrix stored for this run." />
          ) : (
            <ConfusionMatrix rows={confusion.data!.confusion_matrix} />
          )}
        </Card>

        {/* per-class */}
        <Card title="Per-class metrics" sub="From the run's metrics artifact">
          {Object.keys(perClass).length === 0 ? (
            <EmptyState message="Data unavailable" hint="Per-class metrics are only stored for evaluated runs." />
          ) : (
            <div className="scroll-x">
              <table className="table">
                <thead>
                  <tr><th>Activity</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Support</th></tr>
                </thead>
                <tbody>
                  {Object.entries(perClass).map(([label, v]) => (
                    <tr key={label}>
                      <td>{prettyLabel(label)}</td>
                      <td className="num">{fmtPct(v.precision)}</td>
                      <td className="num">{fmtPct(v.recall)}</td>
                      <td className="num">{fmtPct(v.f1)}</td>
                      <td className="num">{v.support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {m?.branch_latency_ms && (
            <div style={{ marginTop: 12 }}>
              <div className="hint" style={{ marginBottom: 6 }}>Branch latency (mean ms per window)</div>
              <div className="row">
                {Object.entries(m.branch_latency_ms).map(([k, v]) => {
                  const name: Record<string, string> = {
                    TDA_MS: "TDA branch", DEEP_MS: "Deep branch", FUSION_MS: "Fusion",
                    tda_ms: "TDA branch", deep_ms: "Deep branch", fusion_ms: "Fusion",
                  };
                  return <Badge key={k} tone="violet">{name[k] ?? prettyLabel(k)}: {fmtNum(Number(v), 2)} ms</Badge>;
                })}
              </div>
            </div>
          )}
        </Card>
      </div>

      {/* training curves */}
      <div className="grid grid-c2" style={{ marginTop: 0 }}>
        <Card title="Training history" sub="Per-epoch loss (train / validation)">
          {(training.data?.epochs?.length ?? 0) === 0 ? (
            <EmptyState message="Data unavailable" hint="Only deep-learning runs log per-epoch history." />
          ) : (
            <LineChart
              series={[
                { name: "train loss", color: "var(--cyan)", points: training.data!.epochs.map((e) => ({ x: e.epoch, y: e.train_loss })) },
                ...(training.data!.epochs.some((e) => e.val_loss != null)
                  ? [{ name: "val loss", color: "var(--pink)", points: training.data!.epochs.filter((e) => e.val_loss != null).map((e) => ({ x: e.epoch, y: e.val_loss as number })), dashed: true }]
                  : []),
              ]}
              height={220}
              xLabel="epoch"
              yLabel="loss"
            />
          )}
          {(training.data?.epochs?.length ?? 0) > 0 && (
            <div style={{ marginTop: 8 }}>
              <LineChart
                series={[
                  { name: "train accuracy", color: "var(--green)", points: training.data!.epochs.map((e) => ({ x: e.epoch, y: e.train_accuracy })) },
                  ...(training.data!.epochs.some((e) => e.val_accuracy != null)
                    ? [{ name: "val accuracy", color: "var(--amber)", points: training.data!.epochs.filter((e) => e.val_accuracy != null).map((e) => ({ x: e.epoch, y: e.val_accuracy as number })), dashed: true }]
                    : []),
                ]}
                height={180}
                xLabel="epoch"
                yLabel="accuracy"
                yDomain={[0, 1]}
              />
            </div>
          )}
        </Card>

        {/* split meta / subject-aware splitting */}
        <Card title="Split & leakage record" sub="Subject-aware splitting proof from the backend store">
          {(detail.data?.split?.length ?? 0) === 0 ? (
            <EmptyState message="Data unavailable" hint="Split metadata was not recorded for this run." />
          ) : (
            <>
              {detail.data!.split.map((row) => (
                <div key={row.fold} className="card tight" style={{ marginBottom: 10 }}>
                  <div className="row" style={{ marginBottom: 8 }}>
                    <Badge tone="cyan">fold {row.fold}</Badge>
                    <Badge tone={row.leakage_passed ? "green" : "red"}>
                      {row.leakage_passed ? "✓ leakage check passed" : "✗ leakage check failed"}
                    </Badge>
                  </div>
                  <div className="kv">
                    <dt>strategy</dt><dd>{typeof row.split_strategy === "string" ? row.split_strategy : "StrategicGroupKFold (subject groups)"}</dd>
                    <dt>train</dt><dd>{row.train_subjects.length} subjects</dd>
                    <dt>val</dt><dd>{row.val_subjects.length} subjects</dd>
                    <dt>test</dt><dd>{row.test_subjects.length} subjects</dd>
                  </div>
                  <div className="hint" style={{ marginTop: 6 }}>
                    test subjects: {row.test_subjects.slice(0, 6).join(", ")}{row.test_subjects.length > 6 ? " …" : ""}
                  </div>
                </div>
              ))}
              <SectionNote tone="green">
                <span><b>Subject-aware splitting.</b> Data is split by subject to prevent overlapping
                windows from the same subject appearing across train and test groups.</span>
              </SectionNote>
            </>
          )}
        </Card>
      </div>

      {/* resources */}
      {(training.data?.resources?.length ?? 0) > 0 && (
        <Card title="Resource measurements" sub="Phase-level RSS / GPU measurements recorded during the run" style={{ marginTop: 14 }}>
          <div className="scroll-x">
            <table className="table">
              <thead>
                <tr><th>Phase</th><th>Metric</th><th className="num">Value</th><th>Recorded</th></tr>
              </thead>
              <tbody>
                {training.data!.resources.slice(0, 24).map((r, i) => (
                  <tr key={i}>
                    <td>{r.phase}</td>
                    <td>{r.metric}</td>
                    <td className="num">{fmtNum(r.value, 2)}</td>
                    <td>{fmtTime(r.timestamp)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* window log */}
      <Card
        title="Per-window prediction log"
        sub="First evaluation-log rows for this run (one row per evaluated window)"
        style={{ marginTop: 14 }}
      >
        {history.loading ? (
          <Skeleton h={220} />
        ) : history.error ? (
          <ErrorState error={history.error} context="Window history" retry={history.reload} />
        ) : (history.data?.windows?.length ?? 0) === 0 ? (
          <EmptyState message="Data unavailable" hint="No per-window evaluation log stored for this run." />
        ) : (
          <div style={{ maxHeight: 340, overflowY: "auto" }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Win</th><th>Subject</th><th>Truth</th><th>Predicted</th>
                  <th className="num">Conf</th><th className="num">ms</th><th className="num">TDA</th><th className="num">Deep</th><th className="num">Fusion</th><th />
                </tr>
              </thead>
              <tbody>
                {history.data!.windows.map((w) => (
                  <tr key={w.id}>
                    <td>{w.window_id}</td>
                    <td>{w.subject_id}</td>
                    <td>{w.true_label ? prettyLabel(w.true_label) : "—"}</td>
                    <td>{prettyLabel(w.predicted_label)}</td>
                    <td className="num">{fmtPct(w.confidence)}</td>
                    <td className="num">{fmtNum(w.inference_time_ms, 1)}</td>
                    <td className="num">{w.tda_ms != null ? fmtNum(w.tda_ms, 1) : "—"}</td>
                    <td className="num">{w.deep_ms != null ? fmtNum(w.deep_ms, 1) : "—"}</td>
                    <td className="num">{w.fusion_ms != null ? fmtNum(w.fusion_ms, 2) : "—"}</td>
                    <td style={{ color: w.correct === 1 ? "var(--green)" : w.correct === 0 ? "var(--red)" : "var(--muted)" }}>
                      {w.correct === 1 ? "✓" : w.correct === 0 ? "✗" : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {selectedRun && (
        <div className="hint" style={{ marginTop: 10 }}>
          Aggregate evaluation (stored metrics) vs per-window prediction (log above) are different
          artifacts — the table is the raw evaluation log, the headline cards are the aggregated
          {selectedRun.model === "svm" ? " SVM baseline" : " model"} results for fold {selectedRun.fold}.
        </div>
      )}
    </>
  );
}
