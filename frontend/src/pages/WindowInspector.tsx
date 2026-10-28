import { useEffect, useMemo, useState } from "react";
import { useAppState } from "../state/AppState";
import { Badge, Button, Card, ErrorState, Field, LoadingBlock, SectionNote, Select, StatCard } from "../components/ui";
import { DatasetSubjectPicker, RunPicker, modelLabel } from "../components/pickers";
import { useInspection } from "../components/useInspection";
import { LineChart } from "../components/charts/LineChart";
import { ScatterPlot, timeColor } from "../components/charts/ScatterPlot";
import { ProbabilityBars } from "../components/charts/ProbabilityBars";
import { fmtMs, fmtNum, fmtS, prettyLabel } from "../lib/format";

const CHANNEL_COLORS = ["#22d3ee", "#a78bfa", "#34d399", "#fbbf24", "#f472b6", "#60a5fa", "#fb923c", "#f87171", "#4ade80"];

export function SensorSeriesChart({
  series,
  channelNames,
  durationS,
}: {
  series: Record<string, number[]>;
  channelNames: string[];
  durationS: number;
}) {
  const names = Object.keys(series);
  const [visible, setVisible] = useState<Set<string>>(new Set());
  const [overlay, setOverlay] = useState(true);

  useEffect(() => {
    // default: show first 3 channels
    setVisible(new Set(names.slice(0, 3)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [names.join("|")]);

  const n = series[names[0]]?.length ?? 0;
  const dt = n > 0 && durationS > 0 ? durationS / n : 0;

  const lines = useMemo(
    () =>
      names
        .filter((nm) => visible.has(nm))
        .map((nm, i) => ({
          name: nm,
          color: CHANNEL_COLORS[i % CHANNEL_COLORS.length],
          points: series[nm].map((v, j) => ({ x: j * dt, y: v })),
        })),
    [names, visible, series, dt],
  );

  return (
    <div>
      <LineChart series={lines} height={240} xLabel="time (s)" yLabel="standardized value" />
      <div className="row" style={{ marginTop: 8, justifyContent: "space-between" }}>
        <div className="legend">
          {names.map((nm, i) => (
            <span
              key={nm}
              className={`legend-item ${visible.has(nm) ? "on" : ""}`}
              onClick={() => {
                const next = new Set(visible);
                if (next.has(nm)) next.delete(nm);
                else next.add(nm);
                setVisible(next);
              }}
            >
              <span className="swatch" style={{ background: CHANNEL_COLORS[i % CHANNEL_COLORS.length] }} />
              {nm}
            </span>
          ))}
        </div>
        <div className="row" style={{ gap: 6 }}>
          <span className="hint">overlay channels</span>
          <div className={`switch ${overlay ? "on" : ""}`} onClick={() => setOverlay((o) => !o)} />
        </div>
      </div>
      <div className="hint" style={{ marginTop: 6 }}>
        {n} samples · Δt = {dt ? (dt * 1000).toFixed(0) : "?"} ms · values are preprocessed
        (standardized) sensor readings served by the backend
      </div>
    </div>
  );
}

export default function WindowInspector() {
  const { select, inspection, subjects, adapters, experiments } = useInspection();
  const { setSelectedRunId } = useAppState();
  const insp = inspection.data;

  const groundTruth = insp?.ground_truth.label ?? null;
  const pred = insp?.prediction ?? null;

  const cloudPoints = useMemo(() => {
    if (!insp) return [];
    const pts = insp.point_cloud.pcaviz2;
    return pts.map(([x, y], i) => ({
      x,
      y,
      color: timeColor(i, Math.max(1, pts.length - 1)),
      label: `point ${insp.point_cloud.point_indices[i] ?? i}`,
      sub: `window sample ${insp.point_cloud.point_indices[i] ?? i}`,
    }));
  }, [insp]);

  return (
    <>
      <div className="page-head">
        <div>
          <div className="page-title">Window Inspector</div>
          <div className="page-desc">
            Pick a dataset, subject and window index to inspect the exact sensor window,
            its point cloud and — with a run attached — the model's prediction with ground truth.
          </div>
        </div>
      </div>

      <Card title="Window selection">
        <div className="toolbar" style={{ marginBottom: 0 }}>
          <DatasetSubjectPicker
            datasets={adapters.data?.adapters ?? []}
            dataset={select.dataset}
            onDataset={select.setDataset}
            subjects={subjects.data?.subjects ?? []}
            subject={select.subject}
            onSubject={select.setSubject}
            windowIndex={select.windowIndex}
            onWindowIndex={select.setWindowIndex}
          />
          <RunPicker
            experiments={experiments}
            value={select.runId || null}
            onChange={select.setRunId}
            label="Attach run (adds prediction)"
            style={{ minWidth: 240 }}
          />
          <Button variant="primary" size="sm" disabled={inspection.loading} onClick={() => inspection.reload()} style={{ marginBottom: 1 }}>
            {inspection.loading ? "Inspecting…" : "Inspect window"}
          </Button>
        </div>
        <div style={{ marginTop: 8 }}>
          <SectionNote>
            <span><b>Note:</b> window indexes restart per subject — the same index refers to
            different data for different subjects. Invalid indexes return a friendly backend
            error.</span>
          </SectionNote>
        </div>
      </Card>

      {inspection.loading ? (
        <div style={{ marginTop: 14 }}>
          <Card><LoadingBlock label="Computing window + TDA (first request loads and preprocesses the dataset)…" /></Card>
        </div>
      ) : inspection.error ? (
        <div style={{ marginTop: 14 }}>
          <Card title="Window Inspector">
            <ErrorState error={inspection.error} context="Window inspection" retry={inspection.reload} />
          </Card>
        </div>
      ) : insp ? (
        <>
          {/* metadata */}
          <div className="grid grid-c4" style={{ marginTop: 14 }}>
            <StatCard label="Dataset / subject" accent="var(--blue)" value={prettyLabel(insp.dataset)} sub={insp.subject_id} />
            <StatCard label="Window index" accent="var(--cyan)" value={insp.window_index} sub={`shape ${insp.window_shape[0]} × ${insp.window_shape[1]}`} />
            <StatCard label="Points / channels" accent="var(--green)" value={`${insp.point_cloud.n_points} / ${insp.window_shape[1]}`} sub={`point cloud dims: ${insp.point_cloud.n_dims}`} />
            <StatCard label="Sampling / duration" accent="var(--violet)" value={`${fmtNum(insp.sampling_rate, 1)} Hz`} sub={`${fmtS(insp.duration_s)} window`} />
          </div>

          <div className="grid grid-c2" style={{ marginTop: 14 }}>
            <Card title="Sensor series" sub="Complete preprocessed window, one line per channel">
              <SensorSeriesChart series={insp.sensor_series} channelNames={insp.channel_names} durationS={insp.duration_s} />
            </Card>

            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <Card title="Window metadata">
                <div className="kv">
                  <dt>dataset</dt><dd>{insp.dataset}</dd>
                  <dt>subject id</dt><dd>{insp.subject_id}</dd>
                  <dt>window index</dt><dd>{insp.window_index}</dd>
                  <dt>window shape</dt><dd>{insp.window_shape[0]} × {insp.window_shape[1]} (T × C)</dd>
                  <dt>sampling rate</dt><dd>{insp.sampling_rate} Hz</dd>
                  <dt>duration</dt><dd>{fmtS(insp.duration_s)}</dd>
                  <dt>ground truth</dt>
                  <dd>{groundTruth ? prettyLabel(groundTruth) : "Not available"}</dd>
                </div>
              </Card>
              <Card title="TDA timings" sub="Backend-measured, per window (ms)">
                <div className="row">
                  {Object.entries(insp.timings_ms).map(([k, v]) => (
                    <Badge key={k}>{k.replace("_ms", "")}: {fmtMs(v, 2)}</Badge>
                  ))}
                </div>
              </Card>
            </div>
          </div>

          <div className="grid grid-c2" style={{ marginTop: 0 }}>
            <Card
              title="Point cloud (PCA projection)"
              sub={insp.point_cloud.note}
              tools={<Badge tone="violet">full-dim points: {insp.point_cloud.n_dims}</Badge>}
            >
              <ScatterPlot
                points={cloudPoints}
                height={250}
                xLabel="PC 1"
                yLabel="PC 2"
                equalAspect
              />
            </Card>

            <Card
              title="Prediction"
              sub={pred ? `run ${pred.model_run_id} · ${modelLabel(runModelName(experiments, select.runId))}` : "attach a run to predict"}
            >
              {pred ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div className="grid grid-c2" style={{ gap: 10 }}>
                    <div className="stat">
                      <div className="stat-label">Predicted activity</div>
                      <div className="stat-value sm" style={{ color: "var(--accent)" }}>{prettyLabel(pred.predicted_label)}</div>
                    </div>
                    <div className="stat">
                      <div className="stat-label">Confidence</div>
                      <div className="stat-value sm" style={{ color: "var(--green)" }}>{(pred.confidence * 100).toFixed(1)}%</div>
                    </div>
                    <div className="stat">
                      <div className="stat-label">Ground truth</div>
                      <div className="stat-value sm">{pred.true_label ? prettyLabel(pred.true_label) : "Not available"}</div>
                    </div>
                    <div className="stat">
                      <div className="stat-label">Result</div>
                      <div className="stat-value sm">
                        {pred.correct == null
                          ? <span style={{ color: "var(--muted)" }}>Not available</span>
                          : pred.correct ? <span style={{ color: "var(--green)" }}>✓ Correct</span> : <span style={{ color: "var(--red)" }}>✗ Incorrect</span>}
                      </div>
                    </div>
                  </div>
                  <div>
                    <div className="hint" style={{ marginBottom: 6 }}>Class probabilities</div>
                    <ProbabilityBars probabilities={pred.probabilities} highlight={pred.predicted_label} compact />
                  </div>
                  <div className="row">
                    <Badge tone="cyan">inference {fmtMs(pred.inference_time_ms)}</Badge>
                    <Badge tone="violet">tda {fmtMs(pred.tda_ms)}</Badge>
                    <Badge tone="amber">deep {fmtMs(pred.deep_ms)}</Badge>
                    <Badge tone="violet">fusion {fmtMs(pred.fusion_ms)}</Badge>
                  </div>
                  {pred.correct == null && (
                    <SectionNote tone="amber">
                      <span>Unlabeled window — correctness is not defined, and no accuracy is computed
                      from a single prediction.</span>
                    </SectionNote>
                  )}
                </div>
              ) : (
                <SectionNote>
                  <span>Select a trained run above — predictions are computed by the backend on
                  this exact window. Ground truth comes from the dataset labels.</span>
                </SectionNote>
              )}
            </Card>
          </div>
        </>
      ) : null}
    </>
  );
}

function runModelName(experiments: { run_id: string; model: string }[], runId: string | null): string {
  if (!runId) return "no run";
  return experiments.find((e) => e.run_id === runId)?.model ?? "unknown";
}
