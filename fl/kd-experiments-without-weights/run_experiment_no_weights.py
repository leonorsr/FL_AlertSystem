from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kd-experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EXPERIMENT_CATALOG, ExperimentConfig, config_to_dict
from data_utils import build_federated_setup
from modeling import create_model, evaluate_model, set_model_parameters, train_local_model
from metrics import ensure_directory, write_json
from custom_flwr import make_client_fn_no_weights, KDNoWeightsStrategy, require_flwr


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


def run_experiment_no_weights(
    experiment_id: str,
    results_dir: Path,
    mode: str = "soft_labels",
    num_rounds_override: int | None = None,
    local_epochs_override: int | None = None,
    random_seed_override: int | None = None,
    config_override: ExperimentConfig | None = None,
):
    if experiment_id not in EXPERIMENT_CATALOG:
        raise KeyError(f"Unknown experiment_id: {experiment_id}")

    config = config_override or EXPERIMENT_CATALOG[experiment_id]
    if config.experiment_id != experiment_id:
        raise ValueError("config_override experiment_id must match experiment_id.")
    if num_rounds_override is not None:
        config = replace(config, num_rounds=int(num_rounds_override))
    if local_epochs_override is not None:
        config = replace(config, local_epochs=int(local_epochs_override))
    if random_seed_override is not None:
        config = replace(config, random_seed=int(random_seed_override))

    fl = require_flwr()
    from flwr.common import ndarrays_to_parameters

    kd_root = str(Path(__file__).resolve().parents[1] / "kd-experiments")
    no_weights_root = str(Path(__file__).resolve().parent)
    pythonpath_parts = [no_weights_root, kd_root]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    setup = build_federated_setup(config)

    # shared client states to collect metrics from clients after simulation
    client_states: dict = {client_id: {} for client_id in setup.train_clients}

    client_fn = make_client_fn_no_weights(setup, config, mode=mode, client_states=client_states)

    # Strategy that keeps initial parameters unchanged but allows aggregation hook
    class BaseNoWeightsStrategy(fl.server.strategy.FedAvg):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.kd_aggregator = KDNoWeightsStrategy()
            self.latest_kd_payload: dict | None = None
            self.latest_dev_metrics: dict[str, float] = {}
            self.latest_test_metrics: dict[str, float] = {}
            self.latest_local_metrics: dict[str, float] = {}

        def _aggregate_metric_prefix(self, results, prefix: str) -> dict[str, float]:
            aggregated: dict[str, float] = {}
            total_examples = 0
            for _, fit_res in results:
                num_examples = int(getattr(fit_res, "num_examples", 0))
                metrics = getattr(fit_res, "metrics", {})
                total_examples += num_examples
                for key, value in metrics.items():
                    if not key.startswith(prefix):
                        continue
                    if not isinstance(value, (int, float, np.integer, np.floating)):
                        continue
                    value = float(value)
                    if np.isnan(value):
                        continue
                    metric_name = key.removeprefix(prefix)
                    aggregated.setdefault(metric_name, 0.0)
                    aggregated[metric_name] += num_examples * value
            if total_examples <= 0:
                return {}
            return {key: value / total_examples for key, value in aggregated.items()}

        def build_fit_config(self, server_round: int) -> dict:
            fit_config = {
                "server_round": int(server_round),
                "local_epochs": int(config.local_epochs),
            }
            if mode == "soft_labels" and self.latest_kd_payload is not None:
                global_soft_pos = self.latest_kd_payload.get("mean_soft_pos")
                if global_soft_pos is not None:
                    fit_config["global_soft_pos"] = float(global_soft_pos)
            if mode == "hidden_states" and self.latest_kd_payload is not None:
                global_hidden_mean = self.latest_kd_payload.get("hidden_mean")
                if isinstance(global_hidden_mean, list):
                    fit_config["global_hidden_mean"] = json.dumps(global_hidden_mean)
            if mode == "class_prototype" and self.latest_kd_payload is not None:
                global_proto_pos_mean = self.latest_kd_payload.get("proto_pos_mean")
                global_proto_neg_mean = self.latest_kd_payload.get("proto_neg_mean")
                if isinstance(global_proto_pos_mean, list):
                    fit_config["global_proto_pos_mean"] = json.dumps(global_proto_pos_mean)
                if isinstance(global_proto_neg_mean, list):
                    fit_config["global_proto_neg_mean"] = json.dumps(global_proto_neg_mean)
            return fit_config

        def aggregate_fit(self, server_round, results, failures):
            if failures:
                first_failure = failures[0]
                raise RuntimeError(
                    f"Flower reported {len(failures)} client failures in round {server_round}. "
                    f"First failure: {first_failure!r}"
                )
            try:
                self.latest_kd_payload = self.kd_aggregator.aggregate_payloads(
                    results,
                    weighted=bool(config.weighted_aggregation),
                    clustered=bool(config.clustered_aggregation),
                )
                self.latest_dev_metrics = self._aggregate_metric_prefix(results, "dev_")
                self.latest_test_metrics = self._aggregate_metric_prefix(results, "test_")
                self.latest_local_metrics = self._aggregate_metric_prefix(results, "val_")
            except Exception:
                self.latest_kd_payload = None
            return self.initial_parameters, {}

    initial_parameters = ndarrays_to_parameters([])

    strategy = BaseNoWeightsStrategy(
        fraction_fit=config.fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=config.min_fit_clients,
        min_evaluate_clients=0,
        min_available_clients=config.min_available_clients,
        evaluate_fn=None,
        fit_metrics_aggregation_fn=None,
        on_fit_config_fn=lambda server_round: strategy.build_fit_config(server_round),
        initial_parameters=initial_parameters,
    )

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
                    "runtime_env": {
                        "env_vars": {
                            "PYTHONPATH": os.environ["PYTHONPATH"],
                        },
                    },
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
                    f"Ray startup failed on attempt {attempt}/{startup_attempts} for experiment '{config.experiment_id}'. Retrying in {wait_seconds}s..."
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

    # After simulation, evaluate final local models on shared dev/test sets
    dev_rows = []
    test_rows = []
    local_rows = []
    for client_id, state in client_states.items():
        params = state.get("current_local_params")
        if params is None:
            continue

        model = create_model(len(setup.feature_columns), config.model)
        set_model_parameters(model, params, keep_local_head=False)

        dev_metrics = evaluate_model(model, setup.dev_arrays[0], setup.dev_arrays[1])
        test_metrics = evaluate_model(model, setup.test_arrays[0], setup.test_arrays[1])
        local_metrics = state.get("last_val_metrics", {})

        dev_rows.append({"client_id": client_id, **dev_metrics})
        test_rows.append({"client_id": client_id, **test_metrics})
        if local_metrics:
            local_rows.append({"client_id": client_id, **local_metrics})

    fine_tune_rows: list[dict] = []
    if config.final_local_finetune_epochs > 0:
        source_states = [state for state in client_states.values() if state.get("current_local_params") is not None]
        source_params = source_states[0]["current_local_params"] if source_states else None
        if source_params is not None:
            for partition in setup.fine_tune_clients.values():
                if len(partition.x_adapt) == 0 or len(partition.x_eval) == 0:
                    continue
                local_model = create_model(len(setup.feature_columns), config.model)
                set_model_parameters(local_model, source_params, keep_local_head=False)
                local_model, _ = train_local_model(
                    model=local_model,
                    x_train=partition.x_adapt,
                    y_train=partition.y_adapt,
                    x_val=partition.x_eval,
                    y_val=partition.y_eval,
                    config=config,
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

    dev_df = pd.DataFrame(dev_rows) if dev_rows else pd.DataFrame()
    test_df = pd.DataFrame(test_rows) if test_rows else pd.DataFrame()
    local_df = pd.DataFrame(local_rows) if local_rows else pd.DataFrame()
    fine_tune_df = pd.DataFrame(fine_tune_rows) if fine_tune_rows else pd.DataFrame()

    def aggregate_df(df: pd.DataFrame) -> dict:
        return df.mean(numeric_only=True).to_dict() if not df.empty else {}

    dev_summary = aggregate_df(dev_df)
    test_summary = aggregate_df(test_df)
    local_summary = aggregate_df(local_df)
    fine_tune_summary = aggregate_df(fine_tune_df)
    if not dev_summary:
        dev_summary = strategy.latest_dev_metrics
    if not test_summary:
        test_summary = strategy.latest_test_metrics
    if not local_summary:
        local_summary = strategy.latest_local_metrics

    run_dir = ensure_directory(results_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    config_payload = config_to_dict(config)
    config_payload["communication_mode"] = mode
    config_payload["communicates_model_weights"] = False
    write_json(run_dir / "config.json", config_payload)

    # Export standard experiment artifacts matching the existing KD experiment format.
    if hasattr(setup, "client_summary"):
        setup.client_summary.to_csv(run_dir / "client_summary.csv", index=False)
    if not dev_summary:
        dev_summary = {}
    if not test_summary:
        test_summary = {}
    if not local_summary:
        local_summary = {}
    if not fine_tune_summary:
        fine_tune_summary = {}

    pd.DataFrame([dev_summary]).to_csv(run_dir / "dev_metrics.csv", index=False)
    pd.DataFrame([test_summary]).to_csv(run_dir / "test_metrics.csv", index=False)
    pd.DataFrame([local_summary]).to_csv(run_dir / "local_metrics_summary.csv", index=False)
    if fine_tune_rows:
        fine_tune_df.to_csv(run_dir / "fine_tune_metrics_by_client.csv", index=False)

    if not local_df.empty:
        local_df.to_csv(run_dir / "local_metrics_by_client.csv", index=False)
    if not dev_df.empty:
        dev_df.to_csv(run_dir / "dev_metrics_by_client.csv", index=False)
    if not test_df.empty:
        test_df.to_csv(run_dir / "test_metrics_by_client.csv", index=False)

    # History-based local round metrics if available from Flower simulation.
    local_round_metrics_records: list[dict] = []
    metrics_distributed_fit = getattr(history, "metrics_distributed_fit", {}) if history is not None else {}
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
    else:
        # If Flower did not record round metrics, write per-client local validation metrics instead.
        fallback_records = []
        for row in local_rows:
            client_id = row.pop("client_id", None)
            for metric_name, metric_value in row.items():
                fallback_records.append(
                    {
                        "client_id": client_id,
                        "metric": metric_name,
                        "value": float(metric_value) if isinstance(metric_value, (int, float, np.integer, np.floating)) else metric_value,
                    }
                )
        if fallback_records:
            pd.DataFrame(fallback_records).sort_values(["client_id", "metric"]).to_csv(
                run_dir / "local_round_metrics.csv", index=False
            )

    history_payload = {
        "losses_centralized": getattr(history, "losses_centralized", []),
        "metrics_centralized": getattr(history, "metrics_centralized", {}),
        "losses_distributed": getattr(history, "losses_distributed", []),
        "metrics_distributed": getattr(history, "metrics_distributed", {}),
        "metrics_distributed_fit": getattr(history, "metrics_distributed_fit", {}),
        "latest_kd_payload": strategy.latest_kd_payload,
    }
    write_json(run_dir / "history.json", history_payload)

    summary = {
        "experiment_id": config.experiment_id,
        "title": config.title,
        "description": config.description,
        "communication_mode": mode,
        "communicates_model_weights": False,
        "num_train_clients": len(setup.train_clients),
        "feature_count": len(setup.feature_columns),
        "dev_metrics": {k: float(v) for k, v in dev_summary.items()},
        "test_metrics": {k: float(v) for k, v in test_summary.items()},
        "local_metrics_summary": {k: float(v) for k, v in local_summary.items()},
        "fine_tune_summary": {k: float(v) for k, v in fine_tune_summary.items()},
        "latest_kd_payload": strategy.latest_kd_payload or {},
    }
    write_json(run_dir / "run_summary.json", summary)

    return run_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run no-weights KD experiments across channels.")
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENT_CATALOG.keys()))
    parser.add_argument("--mode", default="soft_labels", choices=["soft_labels", "hidden_states", "class_prototype"])
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    if args.results_dir is None:
        results_dir = Path(__file__).resolve().parent / args.experiment / "results"
    else:
        results_dir = Path(args.results_dir)

    if args.runs <= 1:
        run_dir = run_experiment_no_weights(
            args.experiment,
            results_dir,
            mode=args.mode,
            num_rounds_override=args.num_rounds,
            local_epochs_override=args.local_epochs,
            random_seed_override=args.random_seed,
        )
        print(f"Saved run artifacts to: {run_dir}")
        return

    base_seed = args.random_seed if args.random_seed is not None else EXPERIMENT_CATALOG[args.experiment].random_seed
    run_dirs: list[Path] = []
    for run_idx in range(args.runs):
        run_seed = base_seed + run_idx
        print(f"Starting run {run_idx + 1}/{args.runs} with random seed {run_seed}...")
        run_dir = run_experiment_no_weights(
            args.experiment,
            results_dir,
            mode=args.mode,
            num_rounds_override=args.num_rounds,
            local_epochs_override=args.local_epochs,
            random_seed_override=run_seed,
        )
        run_dirs.append(run_dir)
        print(f"Saved run artifacts to: {run_dir}")

    print("Completed repeated runs:")
    for run_dir in run_dirs:
        print(f" - {run_dir}")


if __name__ == "__main__":
    main()
