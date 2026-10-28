import { navigate } from "../state/router";

export interface FlowStage {
  name: string;
  desc: string;
  page: string;
  color: string;
}

export const PIPELINE: FlowStage[] = [
  { name: "Sensor Data", desc: "Raw triaxial signals from the dataset", page: "datasets", color: "var(--blue)" },
  { name: "Preprocessing", desc: "Imputation, filtering, standardization", page: "datasets", color: "var(--cyan)" },
  { name: "Windowing", desc: "Sliding windows → (T, C) tensors", page: "windows", color: "var(--green)" },
  { name: "Point Cloud", desc: "Each time step becomes a point", page: "windows", color: "var(--green)" },
  { name: "TDA", desc: "Vietoris-Rips persistent homology", page: "tda", color: "var(--violet)" },
  { name: "DeepConvLSTM", desc: "Conv + LSTM temporal encoder", page: "model", color: "var(--amber)" },
  { name: "Fusion", desc: "Concatenate TDA + deep features", page: "model", color: "var(--pink)" },
  { name: "Prediction", desc: "Softmax activity + confidence", page: "live", color: "var(--cyan)" },
];

export function PipelineFlow({ compact }: { compact?: boolean }) {
  return (
    <div className="flow-row">
      {PIPELINE.map((stage, i) => (
        <div key={stage.name} style={{ display: "flex", alignItems: "stretch", gap: 8, flex: compact ? "0 0 auto" : undefined }}>
          <div
            className="flow-node"
            style={{ borderColor: `color-mix(in srgb, ${stage.color} 30%, transparent)` }}
            onClick={() => navigate(stage.page)}
            title={`${stage.name} → open ${stage.page}`}
          >
            <div className="fn-idx" style={{ color: stage.color }}>{String(i + 1).padStart(2, "0")}</div>
            <div className="fn-name">{stage.name}</div>
            {!compact && <div className="fn-desc">{stage.desc}</div>}
          </div>
          {i < PIPELINE.length - 1 && (
            <div className="flow-arrow" aria-hidden>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 12h14M13 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
