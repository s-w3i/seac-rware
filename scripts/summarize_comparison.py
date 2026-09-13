#!/usr/bin/env python3
"""Aggregate comparison metrics into a compact CSV and Markdown report."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


DISPLAY = {
    "seac": "SEAC", "seac_gru": "SEAC-GRU",
    "shared_a2c": "Shared-A2C", "shared_a2c_gru": "Shared-A2C-GRU",
}
METRICS = (
    "cycles_per_1000_steps", "mean_cycle_time", "deliveries_per_1000_steps",
    "movement_denied_rate", "conflict_attempt_rate", "deadlock_event_rate",
    "wait_step_ratio", "inference_latency_ms",
    "total_deployed_parameters", "policy_networks",
)
COMMON_CONFIG = {
    "env_name": "rware-custom-5ag-routing-v2",
    "num_env_steps": 20000000,
    "time_limit": 500,
    "episodes_per_eval": 100,
    "monitor_eval_seed": 10000,
    "final_eval_seed": 20000,
}
COMMON_ALGORITHM = {
    "lr": 3e-4, "gamma": .99, "entropy_coef": .01,
    "value_loss_coef": .5, "max_grad_norm": .5,
    "num_processes": 4, "num_steps": 5,
}
METHOD_CONFIG = {
    "seac": ("seac", False),
    "seac_gru": ("seac", True),
    "shared_a2c": ("shared_a2c", False),
    "shared_a2c_gru": ("shared_a2c", True),
}


def records(seed_dir):
    files = sorted(seed_dir.glob("*/models/*/metrics.jsonl"))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one result attempt for {seed_dir}, found {len(files)}"
        )
    launch = files[0].parents[2] / "launch.json"
    if launch.exists() and json.loads(launch.read_text()).get("exit_status") != 0:
        raise ValueError(f"Training attempt did not complete: {launch}")
    return [json.loads(line) for line in files[0].read_text().splitlines()]


def validate_provenance(seed_dir, seed, method):
    files = sorted(seed_dir.glob("*/models/*/provenance.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one provenance record for {seed_dir}")
    provenance = json.loads(files[0].read_text())
    config = provenance.get("resolved_config", {})
    for key, expected in COMMON_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"{seed_dir}: {key} does not match comparison protocol")
    if config.get("seed") != seed:
        raise ValueError(f"{seed_dir}: recorded seed does not match directory")
    algorithm = config.get("algorithm", {})
    for key, expected in COMMON_ALGORITHM.items():
        if algorithm.get(key) != expected:
            raise ValueError(
                f"{seed_dir}: algorithm.{key} does not match comparison protocol"
            )
    trainer, recurrent = METHOD_CONFIG[method]
    if config.get("trainer") != trainer or algorithm.get("recurrent_policy") != recurrent:
        raise ValueError(f"{seed_dir}: trainer/recurrent mode does not match {method}")


def latest_values(seed_dir):
    rows = records(seed_dir)
    result = {}
    for kind in ("compute", "evaluation"):
        selected = [row for row in rows if row.get("kind") == kind and (
            kind != "evaluation" or row.get("suite") == "final"
        )]
        if selected:
            result.update(selected[-1])
    return result


def ci(values):
    mean = statistics.fmean(values)
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(
        len(values), 1.96
    )
    half = critical * statistics.stdev(values) / math.sqrt(len(values)) \
        if len(values) > 1 else 0.0
    return mean, half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("results/comparison"))
    args = parser.parse_args()
    summary = []
    learning = []
    for method, label in DISPLAY.items():
        expected_paths = [args.root / method / "seed_643231229"]
        missing = [str(path) for path in expected_paths if not path.exists()]
        if missing:
            raise ValueError(f"{label} is missing required seeds: {', '.join(missing)}")
        per_seed = [latest_values(path) for path in expected_paths]
        if any(not row for row in per_seed):
            raise ValueError(f"{label} has incomplete final results")
        for seed, seed_dir in enumerate(expected_paths):
            validate_provenance(seed_dir, seed, method)
            for record in records(seed_dir):
                if record.get("kind") == "learning":
                    learning.append({"method": label, "seed": seed, **record})
        row = {"method": label, "seeds": len(per_seed)}
        for metric in METRICS:
            values = [float(item[metric]) for item in per_seed if metric in item]
            if values:
                mean, half = ci(values)
                row[metric] = mean
                row[f"{metric}_ci95"] = half
        summary.append(row)

    args.root.mkdir(parents=True, exist_ok=True)
    columns = ["method", "seeds"] + [name for metric in METRICS
                                      for name in (metric, f"{metric}_ci95")]
    with (args.root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)
    with (args.root / "summary.md").open("w") as handle:
        handle.write("| Method | Seeds | Cycles/1k | Cycle time | Conflict rate | Deadlock rate | Params |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            value = lambda key: f"{row[key]:.4g} ± {row.get(key + '_ci95', 0):.3g}" if key in row else "n/a"
            handle.write(f"| {row['method']} | {row['seeds']} | {value('cycles_per_1000_steps')} | "
                         f"{value('mean_cycle_time')} | {value('conflict_attempt_rate')} | "
                         f"{value('deadlock_event_rate')} | {value('total_deployed_parameters')} |\n")
    learning_columns = sorted(set().union(*(row.keys() for row in learning)))
    with (args.root / "learning_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=learning_columns)
        writer.writeheader()
        writer.writerows(learning)
    print(args.root / "summary.md")


if __name__ == "__main__":
    main()
