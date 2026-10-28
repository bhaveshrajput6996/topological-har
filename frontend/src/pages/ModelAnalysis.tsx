import { api } from "../api/client";
import { useApi } from "../api/hooks";
import type { ExperimentDetailResponse } from "../api/types";
import { useAppState } from "../state/AppState";
import { Badge, Card, ErrorState, SectionNote, Skeleton } from "../components/ui";
import { RunPicker, modelLabel } from "../components/pickers";
import { fmtNum, fmtTime, prettyLabel } from "../lib/format";

function ArchNode({ children, dim }: { children: React.ReactNode; dim?: boolean }) {
  return <div className={`arch-node ${dim ? "dim" : ""}`}>{children}</div>;
}
function ArchArrow() {
  return <div className="arch-arrow">↓</div>;
}

export default function ModelAnalysis() {
  const { experiments, selectedRunId, setSelectedRunId, selectedRun } = useAppState();
  const detail = useApi(
    () => api.get<ExperimentDetailResponse>(`/experiments/${selectedRunId}`),
    [selectedRunId],
    { enabled: !!selectedRunId },
  );
  const exp = detail.data?.experiment;
  const cfg = (exp?.config_json ?? {}) as Record<string, any>;
  const modelCfg = cfg.model ?? {};
  const tdaCfg = cfg.tda ?? {};
  const res = piRes();
  function piRes(): [number, number] {
    const r = tdaCfg?.persistence_image?.resolution;
    return Array.isArray(r) && r.length === 2 ? [Number(r[0]), Number(r[1])] : [20, 20];
  }
  const convChannels: number[] = Array.isArray(modelCfg.conv_channels) ? modelCfg.conv_channels : [64, 128, 128, 256];
  const kernel = modelCfg.conv_kernel_size ?? 5;
  const lstmHidden = modelCfg.lstm_hidden ?? 128;
  const headHidden: number[] = Array.isArray(modelCfg.head_hidden) ? modelCfg.head_hidden : [256, 128];
  const piDim = res[0] * res[1];

  return (
    <>
      <div className="page-head">
        <div>
          <div className="page-title">Model Analysis</div>
          <div className="page-desc">
            Conceptual visualization of the backend model architecture — DeepConvLSTM
            (Ordóñez &amp; Roggen 2016, adapted to 1-D convolutions over time) and the
            TDA + DeepConvLSTM fusion model. This page explains the verified backend; it does not
            re-implement anything.
          </div>
        </div>
        <RunPicker
          experiments={experiments}
          value={selectedRunId}
          onChange={setSelectedRunId}
          label="Run (source of the configuration shown)"
          style={{ minWidth: 280 }}
        />
      </div>

      <div className="grid grid-c2">
        {/* DeepConvLSTM encoder */}
        <Card title="DeepConvLSTM encoder" sub="4 × [Conv1d → BatchNorm → ReLU] → 2 × LSTM → deep representation">
          {detail.loading && !exp ? (
            <Skeleton h={280} />
          ) : detail.error ? (
            <ErrorState error={detail.error} context="Run configuration" retry={detail.reload} />
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 2, maxWidth: 340, margin: "0 auto" }}>
              <ArchNode><b>Input</b> · window (T × C)</ArchNode>
              <ArchArrow />
              {[0, 1, 2, 3].map((i) => (
                <div key={i}>
                  <ArchNode>
                    <b>Conv block {i + 1}</b> · {convChannels[i]} ch · kernel {kernel}
                    <div className="dim">Conv1d → BatchNorm → ReLU</div>
                  </ArchNode>
                  <ArchArrow />
                </div>
              ))}
              <ArchNode><b>LSTM 1</b> · hidden {lstmHidden}</ArchNode>
              <ArchArrow />
              <ArchNode><b>LSTM 2</b> · hidden {lstmHidden}</ArchNode>
              <ArchArrow />
              <ArchNode><b>Deep representation</b> · {lstmHidden}-D (last time step)</ArchNode>
            </div>
          )}
          <div style={{ marginTop: 14 }}>
            <SectionNote tone="amber">
              <span>Convolutions run over the time axis for every channel independently
              (channel-preserving 1-D conv, exactly as implemented in
              <span className="mono"> har/models/deepconvlstm.py</span>).</span>
            </SectionNote>
          </div>
        </Card>

        {/* Fusion */}
        <Card title="TDA + DeepConvLSTM fusion" sub="The proposed two-branch model ('TDA + DeepConvLSTM')">
          {detail.loading && !exp ? (
            <Skeleton h={280} />
          ) : detail.error ? null : (
            <div style={{ maxWidth: 420, margin: "0 auto" }}>
              <div className="arch-split">
                <div className="arch-branch">
                  <div className="arch-branch-title" style={{ color: "var(--violet)" }}>TDA branch</div>
                  <ArchNode><b>Persistence image</b><div className="dim">{res[0]}×{res[1]} grid → {piDim}-D</div></ArchNode>
                  <div className="arch-arrow">+</div>
                  {tdaCfg?.statistical_features ? (
                    <ArchNode>
                      <b>Betti statistics</b><div className="dim">H0/H1 counts · persistence · entropy</div>
                    </ArchNode>
                  ) : null}
                </div>
                <div className="arch-branch">
                  <div className="arch-branch-title" style={{ color: "var(--amber)" }}>Deep branch</div>
                  <ArchNode><b>DeepConvLSTM</b><div className="dim">{lstmHidden}-D deep representation</div></ArchNode>
                </div>
              </div>
              <ArchArrow />
              <ArchNode><b>Concatenation</b> · {piDim + lstmHidden}-D fused vector</ArchNode>
              <ArchArrow />
              {headHidden.map((h, i) => (
                <div key={i}>
                  <ArchNode><b>Dense {h}</b> → ReLU → Dropout<div className="dim">classifier head</div></ArchNode>
                  <ArchArrow />
                </div>
              ))}
              <ArchNode><b>Softmax</b> · activity classes</ArchNode>
            </div>
          )}
          <div style={{ marginTop: 14 }}>
            <SectionNote>
              <span><b>Fusion.</b> The TDA representation and the DeepConvLSTM representation are
              concatenated before final classification — one training pass, two complementary
              views of the same window.</span>
            </SectionNote>
          </div>
        </Card>
      </div>

      {/* stored configuration */}
      {exp && (
        <Card
          title={`Stored configuration — ${exp.run_id}`}
          sub={`Stored in the run's experiment record · created ${fmtTime(exp.created_at)}`}
          style={{ marginTop: 14 }}
        >
          <div className="row" style={{ marginBottom: 12 }}>
            <Badge tone="cyan">{modelLabel(exp.model_name)}</Badge>
            <Badge>{exp.dataset}</Badge>
            <Badge tone="violet">window {fmtNum(exp.window_duration, 2)} s</Badge>
            <Badge tone="violet">{exp.window_size} samples · step {exp.step_size}</Badge>
            <Badge>{exp.sampling_rate} Hz</Badge>
            <Badge tone={exp.tda_enabled ? "violet" : ""}>TDA {exp.tda_enabled ? "on" : "off"}</Badge>
            <Badge tone="amber">TDA points {exp.tda_points}</Badge>
          </div>
          <div className="grid grid-c2">
            <div>
              <div className="hint" style={{ marginBottom: 6 }}>Preprocessing</div>
              <div className="kv">
                <dt>interpolation</dt><dd>{prettyLabel(String(cfg.preprocessing?.interpolation ?? "linear"))}</dd>
                <dt>normalization</dt><dd>{prettyLabel(String(cfg.preprocessing?.normalization ?? "standard"))}</dd>
                <dt>filter</dt><dd>{cfg.preprocessing?.filter?.enabled ? `butterworth · order ${cfg.preprocessing.filter.order}` : "disabled"}</dd>
              </div>
            </div>
            <div>
              <div className="hint" style={{ marginBottom: 6 }}>Training</div>
              <div className="kv">
                <dt>batch size</dt><dd>{String(cfg.training?.batch_size ?? 64)}</dd>
                <dt>learning rate</dt><dd>{String(cfg.training?.learning_rate ?? 0.001)}</dd>
                <dt>epochs</dt><dd>{String(cfg.training?.epochs ?? 50)}</dd>
                <dt>scheduler</dt><dd>{prettyLabel(String(cfg.training?.lr_scheduler ?? "plateau"))}</dd>
                <dt>seed</dt><dd>{String(cfg.project?.seed ?? 42)}</dd>
              </div>
            </div>
          </div>
        </Card>
      )}
    </>
  );
}
