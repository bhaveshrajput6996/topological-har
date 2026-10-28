/**
 * Typed responses for the real TDL-HAR backend (backend/app/api/routes.py).
 * Every type mirrors the actual JSON the FastAPI app returns.
 */

export interface HealthResponse {
  status: string;
  version: string;
  torch_cuda: boolean;
}

export interface DatasetInfo {
  name: string;
  kind: string;
  sampling_rate: number;
  channel_names: string[];
  label_names: string[];
  subjects: string[];
  source_path: string;
  notes: string;
}

export interface DatasetsListResponse {
  adapters: string[];
}

export interface DatasetSubjectsResponse {
  dataset: string;
  subjects: string[];
}

export interface ExperimentRow {
  run_id: string;
  experiment_name: string;
  dataset: string;
  model: string;
  fold: number;
  accuracy: number;
  f1_macro: number;
  f1_weighted: number;
  avg_inference_ms: number;
  training_time_s: number;
  peak_ram_mb: number;
  num_windows: number;
  created_at: string;
}

export interface ExperimentsResponse {
  experiments: ExperimentRow[];
}

export interface ExperimentRecord {
  run_id: string;
  experiment_name: string;
  dataset: string;
  model_name: string;
  window_duration: number;
  overlap: number | null;
  window_size: number;
  step_size: number;
  sampling_rate: number;
  tda_enabled: number;
  tda_points: number;
  tda_resolution: string;
  created_at: string;
  git_commit: string | null;
  config_json: Record<string, unknown> | null;
}

export interface EvalSummary {
  run_id: string;
  model_name: string;
  accuracy: number | null;
  precision_macro: number | null;
  recall_macro: number | null;
  f1_macro: number | null;
  precision_weighted: number | null;
  recall_weighted: number | null;
  f1_weighted: number | null;
  avg_inference_ms: number | null;
  p95_inference_ms: number | null;
  peak_ram_mb: number | null;
  training_time_s: number | null;
  num_windows: number | null;
  fold: number | null;
  train_subjects: string[] | string | null;
  val_subjects: string[] | string | null;
  test_subjects: string[] | string | null;
  created_at: string;
}

export interface SplitMetaRow {
  fold: number;
  split_strategy: Record<string, unknown> | null;
  train_subjects: string[];
  val_subjects: string[];
  test_subjects: string[];
  leakage_passed: number;
}

export interface ExperimentDetailResponse {
  run_id: string;
  experiment: ExperimentRecord;
  summary: EvalSummary | null;
  split: SplitMetaRow[];
}

export interface PerClassMetrics {
  [label: string]: { precision: number; recall: number; f1: number; support: number };
}

export interface MetricsJson {
  run_id: string;
  fold: number;
  metrics: {
    accuracy: number;
    precision_macro: number;
    recall_macro: number;
    f1_macro: number;
    precision_weighted: number;
    recall_weighted: number;
    f1_weighted: number;
    per_class: PerClassMetrics;
    num_windows: number;
    inference_latency_ms: {
      mean_ms: number;
      median_ms: number;
      p95_ms: number;
      p99_ms: number;
      max_ms: number;
    };
    branch_latency_ms?: Record<string, number>;
    [k: string]: unknown;
  };
  [k: string]: unknown;
}

export interface MetricsResponse {
  run_id: string;
  summary: EvalSummary | null;
  metrics: MetricsJson | null;
}

export interface ConfusionRow {
  true_label: string;
  predicted_label: string;
  count: number;
}

export interface ConfusionResponse {
  run_id: string;
  confusion_matrix: ConfusionRow[];
}

export interface WindowHistoryRow {
  id: number;
  window_id: number;
  subject_id: string;
  true_label: string | null;
  predicted_label: string;
  confidence: number;
  correct: number | null;
  inference_time_ms: number;
  tda_ms: number;
  deep_ms: number;
  fusion_ms: number;
  timestamp: string;
}

export interface WindowHistoryResponse {
  run_id: string;
  windows: WindowHistoryRow[];
  count: number;
}

export interface EpochRow {
  epoch: number;
  train_loss: number;
  val_loss: number | null;
  train_accuracy: number;
  val_accuracy: number | null;
  epoch_time_s: number;
}

export interface ResourceRow {
  phase: string;
  metric: string;
  value: number;
  timestamp: string;
}

export interface TrainingHistoryResponse {
  run_id: string;
  epochs: EpochRow[];
  resources: ResourceRow[];
}

export interface PredictResponse {
  run_id: string;
  model: string;
  window_id: number | null;
  predicted_activity: string;
  confidence: number;
  probabilities: Record<string, number>;
  inference_time_ms: number;
  tda_time_ms: number;
  deep_time_ms: number;
  fusion_time_ms: number;
  true_activity: string | null;
  correct: boolean | null;
}

export interface DiagramPoint {
  dimension: number;
  birth: number;
  death: number;
  persistence: number;
}

export interface InspectResponse {
  dataset: string;
  subject_id: string;
  window_index: number;
  window_shape: [number, number];
  sampling_rate: number;
  duration_s: number;
  channel_names: string[];
  sensor_series: Record<string, number[]>;
  ground_truth: { label: string | null; label_idx: number | null };
  point_cloud: {
    n_points: number;
    n_dims: number;
    point_indices: number[];
    pcaviz2: [number, number][];
    note: string;
  };
  persistence_diagram: {
    api: Record<string, [number, number][]>;
    points: DiagramPoint[];
    n_essential: Record<string, number>;
    scale_reference: number;
  };
  persistence_image: {
    resolution: [number, number];
    bandwidth: number;
    vector: number[];
  };
  betti_statistics: Record<string, number>;
  timings_ms: { subsample_ms: number; rips_ms: number; image_ms: number };
  prediction?: {
    predicted_label_idx: number;
    confidence: number;
    probabilities: Record<string, number>;
    inference_time_ms: number;
    tda_ms: number;
    deep_ms: number;
    fusion_ms: number;
    predicted_label: string;
    true_label: string | null;
    correct: boolean | null;
    model_run_id: string;
  };
}

export interface ReplayStartResponse {
  session_id: string;
  dataset: string;
  n_windows: number;
  subjects: string[];
  speed: number;
  source: string;
}

export interface ReplayPrediction {
  predicted_label_idx: number;
  confidence: number;
  probabilities: Record<string, number>;
  inference_time_ms: number;
  tda_ms: number;
  deep_ms: number;
  fusion_ms: number;
  predicted_activity: string;
}

export interface ReplayEvent {
  session_id: string;
  dataset: string;
  source: string;
  index: number;
  window_id: number;
  subject_id: string;
  timestamp: number;
  window_shape: [number, number];
  sensor_sample: number[];
  true_activity: string | null;
  true_label_idx: number | null;
  correct: boolean | null;
  prediction: ReplayPrediction | null;
}

export interface ReplayEventsResponse {
  session_id: string;
  events: ReplayEvent[];
}

export interface ReplayControlResponse {
  session_id: string;
  speed: number;
}

export interface ReplayStopResponse {
  session_id: string;
  stopped: boolean;
}

export interface TrainStatusResponse {
  state: "submitted" | "running" | "done" | "failed";
  runs?: Array<Record<string, unknown>>;
  error?: string;
}

export interface TrainStartResponse {
  experiment_id: string;
  state: string;
}

export type ApiMessage = { event: string; session_id: string; [k: string]: unknown };
