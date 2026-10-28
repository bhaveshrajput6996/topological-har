import { useAppState } from "../state/AppState";
import { Field, Select } from "./ui";
import type { ExperimentRow } from "../api/types";

const MODEL_LABELS: Record<string, string> = {
  tda_deepconvlstm: "TDA + DeepConvLSTM",
  deepconvlstm: "DeepConvLSTM",
  svm: "SVM (engineered)",
  cnn: "CNN",
  lstm: "LSTM",
};

export function modelLabel(m: string): string {
  return MODEL_LABELS[m] ?? m;
}

export function RunPicker({
  experiments,
  value,
  onChange,
  label = "Experiment run",
  filter,
  style,
}: {
  experiments: ExperimentRow[];
  value: string | null;
  onChange: (runId: string) => void;
  label?: string;
  filter?: (e: ExperimentRow) => boolean;
  style?: React.CSSProperties;
}) {
  const rows = filter ? experiments.filter(filter) : experiments;
  return (
    <Field label={label} style={style}>
      <Select
        value={value ?? ""}
        placeholder={rows.length ? "Select run…" : "No runs available"}
        onChange={(v) => onChange(v)}
        options={rows.map((e) => ({
          value: e.run_id,
          label: `${e.run_id} · ${modelLabel(e.model)} · ${e.dataset}`,
        }))}
      />
    </Field>
  );
}

/** Dataset + subject + window index selector wired to /datasets endpoints. */
export function DatasetSubjectPicker({
  datasets,
  dataset,
  onDataset,
  subjects,
  subject,
  onSubject,
  windowIndex,
  onWindowIndex,
  maxIndex,
  datasetLoading,
}: {
  datasets: string[];
  dataset: string;
  onDataset: (v: string) => void;
  subjects: string[];
  subject: string;
  onSubject: (v: string) => void;
  windowIndex: number;
  onWindowIndex: (v: number) => void;
  maxIndex?: number;
  datasetLoading?: boolean;
}) {
  return (
    <div className="row" style={{ alignItems: "flex-end" }}>
      <Field label="Dataset">
        <Select
          value={dataset}
          onChange={onDataset}
          options={datasets.map((d) => ({ value: d, label: d }))}
          disabled={datasetLoading}
        />
      </Field>
      <Field label="Subject">
        <Select
          value={subject}
          onChange={onSubject}
          placeholder="All subjects"
          options={subjects.map((s) => ({ value: s, label: s }))}
        />
      </Field>
      <Field label="Window index">
        <input
          className="input mono"
          type="number"
          min={0}
          value={windowIndex}
          onChange={(e) => onWindowIndex(Math.max(0, Math.floor(Number(e.target.value) || 0)))}
          style={{ width: 110 }}
        />
      </Field>
      {maxIndex !== undefined && maxIndex > 0 && (
        <div className="hint mono" style={{ paddingBottom: 8 }}>
          0–{maxIndex}
        </div>
      )}
    </div>
  );
}
