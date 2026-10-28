# TDL-HAR backend

Human Activity Recognition (HAR) backend that fuses a topological-data-analysis
branch (Vietoris-Rips persistent homology on the motion point cloud) into a
DeepConvLSTM learner, with a scikit-learn SVM baseline for honest comparison.
FastAPI app exposes datasets,subjects/window inspection, experiments, per-window
prediction, replay, and profiling.

## Layout

```
backend/
  app/            FastAPI app factory, API routes, services (Replay/Translate)
  har/            main package
    datasets/     UCI HAR (inertial & engineered), synthetic
    preprocessing/ missing-data handling,band-pass filtering,normalisation/scaling
    windowing/    sliding-window (continuous) + dataset-presegmented handling
    subject_split StrategicGroupKFold folds w/subject abstraction & val carve
    tda/          projection image,VR persistence engine (gudhi),Betti stats
    models/       DeepConvLSTM, fusion model, SVM baseline, artifact loader
    training/     Trainer loop (torch), window torch dataset
    inference/    Predictor (fused & single-branch), replay services
    experiments/  ExperimentRunner (multi-fold),inspection & replay CLI,reevaluation
    profiling/    TDA latency benchmark
    db/           SQLite experiment store with runner summary
    config/       Strict YAML config -> dataclasses schema + loaded run config
    cli.py        python -m har_backend {train,evaluate,inspect,benchmark,replay,compare,...}
  configs/        Every experiment YAML (fusion/deepconvlstm/svm/smoke variants)
  scripts/        download_uci_har.py (fetches & extracts UCI HAR zip)
  tests/          pytest suite (73 tests: datasets,splitting,windowing,preproc,
                  TDA,models,training,sqlite,API integration, replay)
  artifacts/      per-run artifact dirs (model.pt/.pkl, config.yaml, metrics)
                  + har_experiments.db / test_runner.db 
```

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r backend/requirements.txt
cd backend
python scripts/download_uci_har.py            # puts UCI HAR under data/UCI HAR Dataset
```

Run the test suite first to confirm everything is wired:

```
cd backend
.venv\Scripts\python.exe -m pytest tests -q     # 73 passed expected
```

## CLI

```
python -m har_backend train      --config configs/uci_fusion.yaml      # multi-fold TDA+DCL training
python -m har_backend train      --config configs/uci_svm.yaml         # SVM baseline (engineered features)
python -m har_backend compare                                          # summary table of all runs
python -m har_backend benchmark  --config configs/uci_fusion.yaml      # TDA latency per window
python -m har_backend inspect    --dataset uci_har --window-index 5    # window + PD + image dump
python -m har_backend replay     --dataset uci_har --run-id HAR-2026-... # terminal replay
python -m har_backend evaluate   --run-id HAR-2026-...                 # re-evaluate stored run
```

## API (uvicorn)

```
cd backend
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

- `GET  /health`, `GET /datasets`, `GET /datasets/{name}/subjects`
- `POST /datasets/load` -> returns dataset meta, subjects & sampling rate
- `GET  /windows/inspect?dataset=&subject=&window_index=` -> sensor series,
  point cloud, persistence diagram/image, Betti stats (+ prediction when
  `run_id` is given)
- `POST /experiments` -> multi-fold run with the posted/YAML config;
  `GET /experiments`, `GET /experiments/{run_id}`, `GET .../confusion-matrix`,
  `GET .../history`, `GET .../window-evaluations`
- `POST /predict` -> alternating inline `data (T,C)` windows, or
  `dataset + subject + window_index` against a stored run
- Replay: `POST /replay/start`, `POST /replay/step`, `POST /replay/play`,
  `GET /replay/{session}/events`, WebSocket `/ws/replay/{session}`,
  `POST /replay/stop`
- CORS open for local frontend (allow_origins=\*).

## Experiments

| run | model | fold | acc | f1_m | latency/window |
|-----|-------|------|-----|------|----------------|
| HAR-2026-003 smoke (800 windows, 2 subjects) | tda_deepconvlstm | 0 | 0.372 | 0.305 | 36.9 ms |
| HAR-2026-007..011 full 5-fold SVM | svm | 0-4 | mean 0.933 | 0.930 | ~3.9 ms |

Smoke run is intentionally tiny (2 subjects / 800 windows / 6 epochs) to prove
the fused pipeline end-to-end; full fused runs should be launched with
`configs/uci_fusion.yaml` (5-fold, all windows) when time allows (~30+ min/fold
dominated by the VR loop on CPU).

## Honest-measurement notes

- Scaler/normalisation are fit on **training windows only** for every fold
  (verified by tests/test_preprocessing.py).
- Splits are subject-grouped (StratifiedGroupKFold over subject ids); no
  subject appears in both train and test sets (tests/test_no_subject_leakage.py).
- TDA uses only the 9 inertial channels; runtime scales with window size and
  point-cloud cap (see `har_backend benchmark`).
- UCI HAR ships rows **pre-cut** (2.56 s / 128 samples, 50% overlap) — overlap
  is always reported as `dataset-defined`; you cannot honestly re-cut those rows
  with a custom overlap (windowing refuses).
