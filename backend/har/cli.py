"""Command line interface of the TDL-HAR backend.

    python -m har_backend train      --config configs/uci_tda.yaml
    python -m har_backend evaluate   --run-id HAR-2026-001
    python -m har_backend replay     --dataset uci_har --run-id HAR-2026-001
    python -m har_backend benchmark  --dataset uci_har
    python -m har_backend inspect-window --dataset uci_har --subject subject_01 --window-index 0 --run-id HAR-2026-001
    python -m har_backend compare
    python -m har_backend serve      --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config.schema import Config, load_config
from .utils.logging import get_logger, log_structured

logger = get_logger("har.cli")


def _train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.fold is not None:
        cfg.evaluation.folds = [args.fold]
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.max_windows:
        cfg.window["default"].max_windows = args.max_windows
    from .experiments.runner import ExperimentRunner

    runner = ExperimentRunner(cfg, experiment_name=args.name or Path(args.config).stem)
    results = runner.run()
    print(json.dumps([r.as_dict() for r in results], indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    """Re-evaluate a stored run on the SAME test subjects it was trained on."""
    from .experiments.reevaluation import evaluate_stored_run

    out = evaluate_stored_run(args.run_id)
    print(json.dumps(out, indent=2)[:4000])
    return 0


def _compare(args: argparse.Namespace) -> int:
    from .db import ExperimentStore

    store = ExperimentStore(_default_db())
    rows = store.summary_rows()
    header = (
        f"{'run_id':<14} {'fold':<4} {'model':<18} {'dataset':<12} "
        f"{'acc':>7} {'f1m':>7} {'f1w':>7} {'avg_ms':>7} {'train_s':>9} {'peakRAM':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.run_id:<14} {row.fold:<4} {row.model_name:<18} {row.dataset:<12} "
            f"{row.accuracy:>7.4f} {row.f1_macro:>7.4f} {row.f1_weighted:>7.4f} "
            f"{row.avg_inference_ms:>7.2f} {row.training_time_s:>9.1f} {row.peak_ram_mb:>8.0f}"
        )
    if not rows:
        print("(no experiments recorded yet -- run `train` first)")
    return 0


def _inspect_window(args: argparse.Namespace) -> int:
    from .experiments.inspection import inspect_dataset_window

    out = inspect_dataset_window(
        dataset_name=args.dataset,
        subject=args.subject,
        window_index=args.window_index,
        run_id=args.run_id,
        config_path=args.config,
    )
    print(json.dumps(out, indent=2)[:6000])
    return 0


def _replay(args: argparse.Namespace) -> int:
    """Run replay in the terminal (REST/WebSocket replay available via API)."""
    from .experiments.replay_cli import run_replay_cli

    run_replay_cli(run_id=args.run_id, dataset_name=args.dataset, limit=args.limit, speed=args.speed)
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from .profiling.tda_benchmark import benchmark_tda
    from .experiments.inspection import representative_window

    window = representative_window(
        dataset_name=args.dataset, config_path=args.config
    )
    out = benchmark_tda(
        window,
        points_list=args.points,
        repetitions=args.repetitions,
        max_features=args.max_features,
    )
    print(json.dumps(out, indent=2))
    if args.save:
        outdir = Path(args.save)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "benchmark.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8"
        )
        print(f"saved -> {outdir / 'benchmark.json'}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _default_db() -> str:
    from .utils.seeds import project_root

    return str(project_root() / "artifacts" / "har_experiments.db")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="har_backend", description="TDL HAR backend CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a model on one dataset + folds")
    p_train.add_argument("--config", required=True, help="YAML configuration file")
    p_train.add_argument("--name", default=None, help="experiment name override")
    p_train.add_argument("--fold", type=int, default=None, help="run just this CV fold")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--max-windows", type=int, default=None)
    p_train.set_defaults(func=_train)

    p_eval = sub.add_parser("evaluate", help="re-evaluate a stored run")
    p_eval.add_argument("--run-id", required=True)
    p_eval.set_defaults(func=_evaluate)

    p_cmp = sub.add_parser("compare", help="table of all stored experiments")
    p_cmp.set_defaults(func=_compare)

    p_insp = sub.add_parser("inspect-window", help="window sensor/TDA inspection")
    p_insp.add_argument("--dataset", default=None)
    p_insp.add_argument("--subject", default=None)
    p_insp.add_argument("--window-index", type=int, default=0)
    p_insp.add_argument("--run-id", default=None, help="model for prediction overlay")
    p_insp.add_argument("--config", default=None, help="config override for dataset/TDA")
    p_insp.set_defaults(func=_inspect_window)

    p_replay = sub.add_parser("replay", help="dataset replay (terminal mode)")
    p_replay.add_argument("--run-id", default=None)
    p_replay.add_argument("--dataset", default=None)
    p_replay.add_argument("--limit", type=int, default=25)
    p_replay.add_argument("--speed", type=float, default=4.0)
    p_replay.set_defaults(func=_replay)

    p_bench = sub.add_parser("benchmark", help="TDA points/latency/RAM benchmark")
    p_bench.add_argument("--dataset", default="synthetic")
    p_bench.add_argument("--config", default=None)
    p_bench.add_argument("--points", type=int, nargs="+", default=[40, 64, 80, 100])
    p_bench.add_argument("--repetitions", type=int, default=3)
    p_bench.add_argument("--max-features", type=int, default=64)
    p_bench.add_argument("--save", default=None)
    p_bench.set_defaults(func=_benchmark)

    p_serve = sub.add_parser("serve", help="start the FastAPI backend")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
