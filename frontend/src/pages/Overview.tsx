import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { useApi } from "../api/hooks";
import type { DatasetInfo, DatasetSubjectsResponse, ExperimentDetailResponse, PredictResponse } from "../api/types";
import { useAppState } from "../state/AppState";
import { Badge, Button, Card, ErrorState, SectionNote, StatCard, StatusDot } from "../components/ui";
import { PipelineFlow } from "../components/PipelineFlow";
import { modelLabel } from "../components/pickers";
import { fmtMs, fmtNum, fmtPct, fmtS, fmtTime, metricOrNull, prettyLabel } from "../lib/format";
import { navigate } from "../state/router";

export default function Overview() {
  const {
    backendStatus, health, lastHealthCheck, experiments, experimentsLoading,
    selectedRun, selectedRunId, setSelectedRunId, selectedDataset, setSelectedDataset, refreshExperiments,
  } = useAppState();

  const detail = useApi(
    () => api.get<ExperimentDetailResponse>(`/experiments/${selectedRunId}`),
    [selectedRunId],
    { enabled: !!selectedRunId },
  );

  const datasetName = selectedDataset ?? selectedRun?.dataset ?? null;
  useEffect(() => {
    if (!selectedDataset && selectedRun) setSelectedDataset(selectedRun.dataset);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRun?.run_id]);

  const datasetInfo = useApi(
    () => api.post<DatasetInfo>("/datasets/load", { name: datasetName }),
    [datasetName],
    { enabled: !!datasetName },
  );

  const subjects = useApi(
    () => api.get<DatasetSubjectsResponse>(`/datasets/${datasetName}/subjects`),
    [datasetName],
    { enabled: !!datasetName },
  );

  // quick prediction: first subject / window 0 of the selected dataset + run
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [predError, setPredError] = useState<string | null>(null);

  const subjectForPredict = useMemo(
    () => subjects.data?.subjects[0] ?? null,
    [subjects.data],
  );

  async function runQuickPredict() {
    if (!selectedRunId || !datasetName || !subjectForPredict) return;
    setPredicting(true);
    setPredError(null);
    try {
      const p = await api.post<PredictResponse>("/predict", {
        run_id: selectedRunId,
        dataset: datasetName,
        subject: subjectForPredict,
        window_index: 0,
      }, 120000);
      setPrediction(p);
    } catch (e) {
      setPrediction(null);
      setPredError((e as Error).message);
    } finally {
      setPredicting(false);
    }
  }

  const detailExp = detail.data?.experiment ?? null;
  const winDur = detailExp?.window_duration ?? null;
  const winSize = detailExp?.window_size ?? null;
  const step = detailExp?.step_size ?? null;
  const sr = detailExp?.sampling_rate ?? datasetInfo.data?.sampling_rate ?? null;
  const tdaOn = detailExp ? !!detailExp.tda_enabled : null;

  const bestPerModel = useMemo(() => {
    const byModel = new Map<string, { row: (typeof experiments)[number]; f1: number | null }>();
    for (const e of experiments) {
      const acc = metricOrNull(e.accuracy);
      if (acc === null) continue;
      const key = e.model;
      const cur = byModel.get(key);
      if (!cur || (metricOrNull(e.f1_macro) ?? -1) > (cur.f1 ?? -1)) byModel.set(key, { row: e, f1: metricOrNull(e.f1_macro) });
    }
    return [...byModel.values()].sort((a, b) => (b.f1 ?? -1) - (a.f1 ?? -1));
  }, [experiments]);

  return (
    <>
      <div className="page-head">
        <div>
          <div className="page-title">Overview</div>
          <div className="page-desc">
            Research platform for Human Activity Recognition combining Vietoris–Rips persistent
            homology (TDA) with a DeepConvLSTM temporal encoder. All metrics on this page are read
            live from the backend.
          </div>
        </div>
        <div className="row" style={{ marginLeft: "auto" }}>
          <Badge tone={backendStatus === "online" ? "green" : "red"}>
            <StatusDot status={backendStatus === "online" ? "ok" : "err"} pulse={backendStatus === "online"} />
            backend {backendStatus}
          </Badge>
          {health && <Badge>{health.torch_cuda ? "CUDA" : "CPU runtime"}</Badge>}
          <Button size="sm" onClick={() => void refreshExperiments()}>Refresh data</Button>
        </div>
      </div>

      {/* pipeline */}
      <Card title="Research pipeline" sub="Each stage is computed by the backend — click a stage to inspect it">
        <PipelineFlow />
      </Card>

      {/* stat cards */}
      <div className="grid grid-c4" style={{ marginTop: 14 }}>
        <StatCard
          label="Active dataset"
          accent="var(--blue)"
          loading={experimentsLoading}
          value={datasetName ? prettyLabel(datasetName) : "—"}
          sub={datasetInfo.data
            ? `${datasetInfo.data.subjects.length} subjects · ${datasetInfo.data.label_names.length} activities · ${datasetInfo.data.sampling_rate} Hz`
            : datasetName ? "loading…" : "select a dataset"}
        />
        <StatCard
          label="Active model"
          accent="var(--amber)"
          value={selectedRun ? modelLabel(selectedRun.model) : "—"}
          sub={selectedRun
            ? `${selectedRun.run_id} · ${selectedRun.experiment_name} · fold ${selectedRun.fold}`
            : "select a run in Experiments"}
        />
        <StatCard
          label="Current activity (predicted)"
          accent="var(--cyan)"
          loading={predicting}
          value={prediction ? prediction.predicted_activity.replace(/_/g, " ") : "—"}
          sub={prediction
            ? `window ${prediction.window_id ?? 0}${prediction.true_activity ? ` · truth: ${prediction.true_activity.replace(/_/g, "")}` : " · truth: N/A"}`
            : "predict window 0"}
        >
          <div style={{ marginTop: 8, display: "flex", gap: 6 }}>
            <Button size="sm" variant="primary" disabled={!selectedRunId || !subjectForPredict || predicting} onClick={() => void runQuickPredict()}>
              {predicting ? "Predicting…" : "Predict sample"}
            </Button>
          </div>
          {predError && <div className="hint" style={{ color: "var(--red)", marginTop: 6 }}>{predError}</div>}
        </StatCard>
        <StatCard
          label="Prediction confidence"
          accent="var(--green)"
          loading={predicting}
          value={prediction ? fmtPct(prediction.confidence) : "—"}
          sub={prediction ? `inference ${fmtMs(prediction.inference_time_ms)}` : "not run yet"}
        />
      </div>

      <div className="grid grid-c4" style={{ marginTop: 0 }}>
        <StatCard
          label="Window duration"
          accent="var(--green)"
          value={winDur ? `${fmtNum(winDur, 2)} s` : "—"}
          sub={winSize != null && sr ? `${winSize} samples @ ${sr} Hz` : "from run configuration"}
        />
        <StatCard
          label="Sampling rate"
          accent="var(--blue)"
          value={sr ? `${fmtNum(sr, 1)} Hz` : "—"}
          sub={datasetInfo.data?.kind ? `dataset kind: ${datasetInfo.data.kind}` : undefined}
        />
        <StatCard
          label="TDA status"
          accent="var(--violet)"
          value={tdaOn === null ? "—" : tdaOn ? "Enabled" : "Disabled"}
          sub={detailExp?.tda_points ? `${detailExp.tda_points} TDA points · image ${detailExp.tda_resolution}` : "from run configuration"}
        />
        <StatCard
          label="Backend status"
          accent={backendStatus === "online" ? "var(--green)" : "var(--red)"}
          value={backendStatus === "online" ? "Connected" : backendStatus === "offline" ? "Offline" : "Checking"}
          sub={lastHealthCheck ? `last check ${fmtTime(new Date(lastHealthCheck).toISOString())} · v${health?.version ?? "?"}` : "…"}
        >
          <div style={{ marginTop: 8 }}>
            <Button size="sm" onClick={() => navigate("system")}>Diagnostics</Button>
          </div>
        </StatCard>
      </div>

      <div className="grid grid-c2" style={{ marginTop: 14 }}>
        {/* best runs */}
        <Card
          title="Best completed run per model"
          sub="From the backend experiment store (highest F1-macro); missing metrics show N/A"
          tools={<Button size="sm" variant="ghost" onClick={() => navigate("experiments")}>All experiments →</Button>}
        >
          {experimentsLoading ? (
            <div className="skeleton" style={{ height: 120 }} />
          ) : bestPerModel.length === 0 ? (
            <div className="hint" style={{ padding: 12 }}>No completed evaluation runs yet — train a model from the Experiments page.</div>
          ) : (
            <div className="scroll-x">
              <table className="table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Run</th>
                    <th className="num">Accuracy</th>
                    <th className="num">F1 (macro)</th>
                    <th className="num">Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {bestPerModel.map(({ row }) => (
                    <tr
                      key={row.run_id}
                      className="clickable"
                      onClick={() => { setSelectedRunId(row.run_id); navigate("evaluation", { run: row.run_id }); }}
                    >
                      <td>{modelLabel(row.model)}</td>
                      <td>{row.run_id}</td>
                      <td className="num">{fmtPct(metricOrNull(row.accuracy))}</td>
                      <td className="num">{fmtPct(metricOrNull(row.f1_macro))}</td>
                      <td className="num">{row.avg_inference_ms > 0 ? fmtMs(row.avg_inference_ms) : "N/A"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        {/* transparency */}
        <Card title="Research transparency" sub="Key methodological points, as implemented by the backend">
          <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
            <SectionNote tone="green">
              <span><b>Subject-aware splitting.</b> Data is split by subject (StrategicGroupKFold) so
              overlapping windows from the same subject never appear in both train and test groups.</span>
            </SectionNote>
            <SectionNote tone="violet">
              <span><b>TDA.</b> Persistent homology (H0/H1) is computed on the full-dimensional
              standardized point cloud, not on a reduced projection.</span>
            </SectionNote>
            <SectionNote>
              <span><b>PCA.</b> PCA appears only in visualizations — it is never the TDA input.</span>
            </SectionNote>
            <SectionNote tone="amber">
              <span><b>Fusion.</b> The TDA representation and the DeepConvLSTM representation are
              concatenated before the final classifier head.</span>
            </SectionNote>
          </div>
        </Card>
      </div>

      {/* window config from the actual run */}
      {detail.error && <div style={{ marginTop: 14 }}><ErrorState error={detail.error} context="Could not load run configuration" retry={detail.reload} /></div>}
      {detail.data && (
        <Card title={`Selected run configuration — ${detail.data.run_id}`} sub="Displayed values come from the run's stored experiment record" style={{ marginTop: 14 }}>
          <div className="kv" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(150px, max-content))", gap: "8px 24px" }}>
            {([
              ["window duration", detailExp ? `${fmtNum(detailExp.window_duration, 2)} s` : "N/A"],
              ["window size", detailExp ? `${detailExp.window_size} samples` : "N/A"],
              ["step size", detailExp ? `${detailExp.step_size} samples` : "N/A"],
              ["overlap", detailExp ? (detailExp.overlap != null ? fmtPct(detailExp.overlap) : "dataset-defined") : "N/A"],
              ["sampling rate", detailExp ? `${detailExp.sampling_rate} Hz` : "N/A"],
              ["TDA", detailExp ? (detailExp.tda_enabled ? "enabled" : "disabled") : "N/A"],
              ["TDA feature dim", (detail.data.summary && String(detail.data.summary.model_name).includes("svm")) ? "N/A (SVM)" : "—"],
              ["num windows (eval)", detail.data.summary?.num_windows ?? "N/A"],
              ["created", fmtTime(detailExp?.created_at)],
            ] as [string, string][]).map(([k, v]) => (
              <div key={k} style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <span className="hint" style={{ textTransform: "uppercase", fontSize: 10, letterSpacing: "0.08em" }}>{k}</span>
                <span className="mono" style={{ fontSize: 12.5 }}>{v}</span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </>
  );
}
