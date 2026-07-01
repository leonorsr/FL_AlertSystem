from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KD_ROOT = ROOT / "kd-experiments"
THIS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(KD_ROOT))
sys.path.insert(0, str(THIS_ROOT))

from binned_probability_payloads import BinnedProbabilityPayloadStrategy, make_binned_payload_client_fn, require_flwr
from bin_payload_configs import BIN_PAYLOAD_CONFIGS, DEFAULT_BIN_PAYLOAD_ORDER
from config import EXPERIMENT_CATALOG, ExperimentConfig, config_to_dict
from data_utils import build_federated_setup
from metrics import ensure_directory, write_json
from modeling import create_model, evaluate_model, set_model_parameters, train_local_model


DEFAULT_RESULTS_ROOT = THIS_ROOT / "results"
DEFAULT_EXPERIMENT_ID = "exp11_baseline_final"


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


def _aggregate_metric_prefix(results, prefix: str) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    total_examples = 0
    for _, fit_res in results:
        num_examples = int(getattr(fit_res, "num_examples", 0))
        metrics = getattr(fit_res, "metrics", {}) or {}
        total_examples += num_examples
        for key, value in metrics.items():
            if not key.startswith(prefix) or not isinstance(value, (int, float, np.integer, np.floating)):
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


def _collect_final_client_metrics(setup, config, client_states: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    return pd.DataFrame(dev_rows), pd.DataFrame(test_rows), pd.DataFrame(local_rows)


def _mean_summary(df: pd.DataFrame) -> dict[str, float]:
    return {key: float(value) for key, value in df.mean(numeric_only=True).to_dict().items()} if not df.empty else {}


def run_binned_probability_payload_experiment(
    payload_name: str,
    results_dir: Path,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    num_rounds_override: int | None = None,
    local_epochs_override: int | None = None,
    random_seed_override: int | None = None,
    config_override: ExperimentConfig | None = None,
) -> Path:
    if payload_name not in BIN_PAYLOAD_CONFIGS:
        raise KeyError(f"Unknown payload_name: {payload_name}")
    if experiment_id not in EXPERIMENT_CATALOG:
        raise KeyError(f"Unknown experiment_id: {experiment_id}")

    config = config_override or EXPERIMENT_CATALOG[experiment_id]
    if num_rounds_override is not None:
        config = replace(config, num_rounds=int(num_rounds_override))
    if local_epochs_override is not None:
        config = replace(config, local_epochs=int(local_epochs_override))
    if random_seed_override is not None:
        config = replace(config, random_seed=int(random_seed_override))

    fl = require_flwr()
    from flwr.common import ndarrays_to_parameters

    pythonpath_parts = [str(THIS_ROOT), str(KD_ROOT)]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    setup = build_federated_setup(config)
    client_states: dict = {client_id: {} for client_id in setup.train_clients}
    client_fn = make_binned_payload_client_fn(setup, config, payload_name=payload_name, client_states=client_states)
    payload_strategy = BinnedProbabilityPayloadStrategy(payload_name)

    class BinnedPayloadFedAvg(fl.server.strategy.FedAvg):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.latest_payload = None
            self.latest_dev_metrics: dict[str, float] = {}
            self.latest_test_metrics: dict[str, float] = {}
            self.latest_local_metrics: dict[str, float] = {}

        def build_fit_config(self, server_round: int) -> dict:
            fit_config = {"server_round": int(server_round), "local_epochs": int(config.local_epochs)}
            if self.latest_payload is not None:
                fit_config["global_probability_payload"] = json.dumps(self.latest_payload, ensure_ascii=True)
            return fit_config

        def aggregate_fit(self, server_round, results, failures):
            if failures:
                raise RuntimeError(f"Flower reported {len(failures)} failures in round {server_round}: {failures[0]!r}")
            self.latest_payload = payload_strategy.aggregate_payloads(results)
            self.latest_dev_metrics = _aggregate_metric_prefix(results, "dev_")
            self.latest_test_metrics = _aggregate_metric_prefix(results, "test_")
            self.latest_local_metrics = _aggregate_metric_prefix(results, "val_")
            return self.initial_parameters, {}

    strategy = BinnedPayloadFedAvg(
        fraction_fit=config.fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=config.min_fit_clients,
        min_evaluate_clients=0,
        min_available_clients=config.min_available_clients,
        evaluate_fn=None,
        fit_metrics_aggregation_fn=None,
        on_fit_config_fn=lambda server_round: strategy.build_fit_config(server_round),
        initial_parameters=ndarrays_to_parameters([]),
    )

    history = None
    last_exception: Exception | None = None
    for attempt in range(1, 4):
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
                    "runtime_env": {"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}},
                },
            )
            break
        except Exception as exc:
            last_exception = exc
            _maybe_shutdown_ray()
            gc.collect()
            if attempt < 3 and _is_ray_startup_timeout(exc):
                time.sleep(5 * attempt)
                continue
            raise
        finally:
            _maybe_shutdown_ray()
    if history is None:
        raise RuntimeError("Flower simulation did not produce a history object.") from last_exception

    dev_df, test_df, local_df = _collect_final_client_metrics(setup, config, client_states)
    dev_summary = _mean_summary(dev_df) or strategy.latest_dev_metrics
    test_summary = _mean_summary(test_df) or strategy.latest_test_metrics
    local_summary = _mean_summary(local_df) or strategy.latest_local_metrics

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
                fine_tune_rows.append({"client_id": partition.client_id, "dataset": partition.dataset, "client": partition.client, **metrics})
    fine_tune_df = pd.DataFrame(fine_tune_rows)
    fine_tune_summary = _mean_summary(fine_tune_df)

    payload_config = BIN_PAYLOAD_CONFIGS[payload_name]
    run_dir = ensure_directory(results_dir / payload_name / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    config_payload = config_to_dict(config)
    config_payload["payload_name"] = payload_name
    config_payload["payload_description"] = payload_config.description
    config_payload["communicates_model_weights"] = False
    config_payload["payload_scalar_count"] = payload_config.scalar_count
    config_payload["num_bins"] = payload_config.num_bins
    write_json(run_dir / "config.json", config_payload)

    if hasattr(setup, "client_summary"):
        setup.client_summary.to_csv(run_dir / "client_summary.csv", index=False)
    pd.DataFrame([dev_summary]).to_csv(run_dir / "dev_metrics.csv", index=False)
    pd.DataFrame([test_summary]).to_csv(run_dir / "test_metrics.csv", index=False)
    pd.DataFrame([local_summary]).to_csv(run_dir / "local_metrics_summary.csv", index=False)
    if not dev_df.empty:
        dev_df.to_csv(run_dir / "dev_metrics_by_client.csv", index=False)
    if not test_df.empty:
        test_df.to_csv(run_dir / "test_metrics_by_client.csv", index=False)
    if not local_df.empty:
        local_df.to_csv(run_dir / "local_metrics_by_client.csv", index=False)
    if not fine_tune_df.empty:
        fine_tune_df.to_csv(run_dir / "fine_tune_metrics_by_client.csv", index=False)

    history_payload = {
        "losses_centralized": getattr(history, "losses_centralized", []),
        "metrics_centralized": getattr(history, "metrics_centralized", {}),
        "losses_distributed": getattr(history, "losses_distributed", []),
        "metrics_distributed": getattr(history, "metrics_distributed", {}),
        "metrics_distributed_fit": getattr(history, "metrics_distributed_fit", {}),
        "latest_probability_payload": strategy.latest_payload,
    }
    write_json(run_dir / "history.json", history_payload)
    summary = {
        "experiment_id": config.experiment_id,
        "title": config.title,
        "payload_name": payload_name,
        "payload_description": payload_config.description,
        "payload_scalar_count": payload_config.scalar_count,
        "num_bins": payload_config.num_bins,
        "communicates_model_weights": False,
        "num_train_clients": len(setup.train_clients),
        "feature_count": len(setup.feature_columns),
        "dev_metrics": {k: float(v) for k, v in dev_summary.items()},
        "test_metrics": {k: float(v) for k, v in test_summary.items()},
        "local_metrics_summary": {k: float(v) for k, v in local_summary.items()},
        "fine_tune_summary": {k: float(v) for k, v in fine_tune_summary.items()},
        "latest_probability_payload": strategy.latest_payload or {},
    }
    write_json(run_dir / "run_summary.json", summary)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run binned soft-label probability payload experiments.")
    parser.add_argument("--payload", choices=DEFAULT_BIN_PAYLOAD_ORDER, required=True)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT_ID, choices=sorted(EXPERIMENT_CATALOG.keys()))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    base_seed = args.random_seed if args.random_seed is not None else EXPERIMENT_CATALOG[args.experiment].random_seed
    for run_idx in range(args.runs):
        run_seed = base_seed + run_idx
        run_dir = run_binned_probability_payload_experiment(
            payload_name=args.payload,
            results_dir=Path(args.results_dir),
            experiment_id=args.experiment,
            num_rounds_override=args.num_rounds,
            local_epochs_override=args.local_epochs,
            random_seed_override=run_seed,
        )
        print(f"Saved run {run_idx + 1}/{args.runs} to: {run_dir}")


if __name__ == "__main__":
    main()
