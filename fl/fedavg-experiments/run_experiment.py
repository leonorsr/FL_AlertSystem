from __future__ import annotations

import argparse
import gc
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_EXPERIMENTS_ROOT, DEFAULT_RESULTS_DIR, EXPERIMENT_CATALOG, config_to_dict, get_experiment_results_dir
from data_utils import build_federated_setup
from flwr_utils import build_final_local_metrics_table, create_strategy, make_client_fn, make_evaluate_fn, require_flwr
from metrics import ensure_directory, write_json
from modeling import create_model, evaluate_model, get_model_parameters, set_model_parameters, train_local_model


def _maybe_shutdown_ray() -> None:
    try:
        import ray
    except ImportError:
        return

    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


def _is_ray_startup_timeout(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "timed out during startup" in message
        or "gcs cannot find the node" in message
        or "failed to get node info" in message
        or "raylet failed to startup" in message
        or "gcs has become overloaded" in message
    )


def run_experiment(experiment_id: str, results_dir: Path) -> Path:
    return run_experiment_with_overrides(experiment_id=experiment_id, results_dir=results_dir)


def run_experiment_with_overrides(
    experiment_id: str,
    results_dir: Path,
    num_rounds_override: int | None = None,
    local_epochs_override: int | None = None,
    random_seed_override: int | None = None,
) -> Path:
    if experiment_id not in EXPERIMENT_CATALOG:
        raise KeyError(f"Unknown experiment_id: {experiment_id}")

    config = EXPERIMENT_CATALOG[experiment_id]
    if num_rounds_override is not None:
        config = replace(config, num_rounds=int(num_rounds_override))
    if local_epochs_override is not None:
        config = replace(config, local_epochs=int(local_epochs_override))
    if random_seed_override is not None:
        config = replace(config, random_seed=int(random_seed_override))

    fl, _, _ = require_flwr()

    setup = build_federated_setup(config)
    base_model = create_model(len(setup.feature_columns), config.model)
    initial_parameters = get_model_parameters(base_model)

    client_fn, client_states = make_client_fn(setup, config)
    strategy = create_strategy(setup, config, initial_parameters)
    evaluate_ndarrays, _ = make_evaluate_fn(setup, config)

    history = None
    startup_attempts = 3
    last_exception: Exception | None = None

    for attempt in range(1, startup_attempts + 1):
        _maybe_shutdown_ray()
        gc.collect()
        try:
            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=len(setup.train_clients),
                config=fl.server.ServerConfig(num_rounds=config.num_rounds),
                strategy=strategy,
                client_resources={"num_cpus": 1},
                ray_init_args={
                    "include_dashboard": False,
                    "ignore_reinit_error": True,
                },
            )
            break
        except Exception as exc:
            last_exception = exc
            _maybe_shutdown_ray()
            gc.collect()
            if attempt < startup_attempts and _is_ray_startup_timeout(exc):
                wait_seconds = 5 * attempt
                print(
                    f"Ray startup failed on attempt {attempt}/{startup_attempts} "
                    f"for experiment '{config.experiment_id}'. Retrying in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)
                continue
            raise
        finally:
            _maybe_shutdown_ray()

    if history is None:
        raise RuntimeError(
            f"Flower simulation did not produce a history object for experiment '{config.experiment_id}'."
        ) from last_exception

    final_model = create_model(len(setup.feature_columns), config.model)
    set_model_parameters(final_model, strategy.latest_parameters_ndarrays, keep_local_head=False)

    dev_metrics = evaluate_ndarrays(config.num_rounds, strategy.latest_parameters_ndarrays, "dev")
    test_metrics = evaluate_ndarrays(config.num_rounds, strategy.latest_parameters_ndarrays, "test")
    local_metrics_by_client_df, local_metrics_summary = build_final_local_metrics_table(setup, config, client_states)
    if local_metrics_by_client_df is None and not getattr(strategy, "latest_local_metrics_by_client", pd.DataFrame()).empty:
        local_metrics_by_client_df = strategy.latest_local_metrics_by_client
        local_metrics_summary = strategy.latest_local_metrics_summary

    fine_tune_rows: list[dict] = []
    if config.final_local_finetune_epochs > 0:
        for partition in setup.fine_tune_clients.values():
            if len(partition.x_adapt) == 0 or len(partition.x_eval) == 0:
                continue
            local_model = create_model(len(setup.feature_columns), config.model)
            set_model_parameters(local_model, strategy.latest_parameters_ndarrays, keep_local_head=False)
            local_model, _ = train_local_model(
                model=local_model,
                x_train=partition.x_adapt,
                y_train=partition.y_adapt,
                x_val=partition.x_eval,
                y_val=partition.y_eval,
                config=config.model,
                local_epochs=config.final_local_finetune_epochs,
                seed=config.random_seed,
            )
            metrics = evaluate_model(local_model, partition.x_eval, partition.y_eval)
            fine_tune_rows.append(
                {
                    "client_id": partition.client_id,
                    "dataset": partition.dataset,
                    "client": partition.client,
                    **metrics,
                }
            )

    run_dir = ensure_directory(results_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    setup.client_summary.to_csv(run_dir / "client_summary.csv", index=False)
    pd.DataFrame([dev_metrics]).to_csv(run_dir / "dev_metrics.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(run_dir / "test_metrics.csv", index=False)
    if local_metrics_by_client_df is not None:
        local_metrics_by_client_df.to_csv(run_dir / "local_metrics_by_client.csv", index=False)
        pd.DataFrame([local_metrics_summary]).to_csv(run_dir / "local_metrics_summary.csv", index=False)

    if fine_tune_rows:
        fine_tune_df = pd.DataFrame(fine_tune_rows).sort_values(["pr_auc", "miss_rate", "far"], ascending=[False, True, True])
        fine_tune_df.to_csv(run_dir / "fine_tune_metrics_by_client.csv", index=False)
        fine_tune_summary = fine_tune_df.mean(numeric_only=True).to_dict()
    else:
        fine_tune_summary = {}

    history_payload = {
        "losses_centralized": getattr(history, "losses_centralized", []),
        "metrics_centralized": getattr(history, "metrics_centralized", {}),
        "losses_distributed": getattr(history, "losses_distributed", []),
        "metrics_distributed": getattr(history, "metrics_distributed", {}),
        "metrics_distributed_fit": getattr(history, "metrics_distributed_fit", {}),
    }
    write_json(run_dir / "history.json", history_payload)

    local_round_metrics_records: list[dict] = []
    metrics_distributed_fit = getattr(history, "metrics_distributed_fit", {})
    for metric_name, entries in metrics_distributed_fit.items():
        for server_round, metric_value in entries:
            local_round_metrics_records.append(
                {
                    "round": int(server_round),
                    "metric": metric_name,
                    "value": float(metric_value) if isinstance(metric_value, (int, float, np.integer, np.floating)) else metric_value,
                }
            )
    if local_round_metrics_records:
        pd.DataFrame(local_round_metrics_records).sort_values(["round", "metric"]).to_csv(
            run_dir / "local_round_metrics.csv", index=False
        )

    summary = {
        "experiment_id": config.experiment_id,
        "title": config.title,
        "description": config.description,
        "num_train_clients": len(setup.train_clients),
        "feature_count": len(setup.feature_columns),
        "dev_metrics": {key: float(value) if isinstance(value, (int, float, np.integer, np.floating)) else value for key, value in dev_metrics.items()},
        "test_metrics": {key: float(value) if isinstance(value, (int, float, np.integer, np.floating)) else value for key, value in test_metrics.items()},
        "local_metrics_summary": {key: float(value) for key, value in local_metrics_summary.items()},
        "fine_tune_summary": {key: float(value) for key, value in fine_tune_summary.items()},
    }
    write_json(run_dir / "config.json", config_to_dict(config))
    write_json(run_dir / "run_summary.json", summary)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Flower-based FedAvg experiments for fall detection.")
    parser.add_argument(
        "--experiment",
        default="exp1_fedavg_base",
        choices=sorted(EXPERIMENT_CATALOG.keys()),
        help="Experiment configuration to run.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Optional directory where run artifacts will be saved. If omitted, results are stored under fedavg-experiments/<experiment_id>/results/.",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=None,
        help="Optional override for the number of global rounds.",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=None,
        help="Optional override for the number of local epochs per client update.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional override for the experiment random seed. Useful for repeated-run uncertainty estimation.",
    )
    args = parser.parse_args()

    if args.results_dir is None:
        results_dir = get_experiment_results_dir(EXPERIMENT_CATALOG[args.experiment], DEFAULT_EXPERIMENTS_ROOT)
    else:
        results_dir = Path(args.results_dir)

    run_dir = run_experiment_with_overrides(
        experiment_id=args.experiment,
        results_dir=results_dir,
        num_rounds_override=args.num_rounds,
        local_epochs_override=args.local_epochs,
        random_seed_override=args.random_seed,
    )
    print(f"Saved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
