#!/usr/bin/env python3
"""Audit Sacred runs for a compatible, complete Phase-2 SEAC baseline."""

import argparse
import json
from pathlib import Path
import tarfile


EXPECTED = {
    "env_name": "rware-custom-5ag-routing-v2",
    "time_limit": 500,
    "num_env_steps": 40000000,
}
EXPECTED_ALGORITHM = {
    "lr": 3e-4, "gamma": 0.99, "entropy_coef": 0.01,
    "value_loss_coef": 0.5, "max_grad_norm": 0.5,
    "num_processes": 4, "num_steps": 5, "recurrent_policy": False,
}
EXPECTED_SEEDS = set(range(5))


def audit(run):
    reasons = []
    try:
        config = json.loads((run / "config.json").read_text())
        metadata = json.loads((run / "run.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable metadata: {error}"]
    for key, expected in EXPECTED.items():
        if config.get(key) != expected:
            reasons.append(f"{key}={config.get(key)!r}, expected {expected!r}")
    for key, expected in EXPECTED_ALGORITHM.items():
        actual = config.get("algorithm", {}).get(key)
        if actual != expected:
            reasons.append(f"algorithm.{key}={actual!r}, expected {expected!r}")
    if not isinstance(config.get("seed"), int):
        reasons.append("integer seed provenance missing")
    elif config["seed"] not in EXPECTED_SEEDS:
        reasons.append(f"seed={config['seed']!r}, expected one of 0..4")
    if config.get("episodes_per_eval", 0) < 100:
        reasons.append("fewer than 100 evaluation episodes")
    if metadata.get("status") != "COMPLETED":
        reasons.append(f"run status is {metadata.get('status')!r}")
    archives = sorted(run.glob("u*.tar.xz"))
    if not archives:
        reasons.append("checkpoint archive missing")
    else:
        expected_update = int(config.get("num_env_steps", 0)) // max(
            1, config.get("algorithm", {}).get("num_steps", 1)
            * config.get("algorithm", {}).get("num_processes", 1)
        )
        if archives[-1].stem != f"u{expected_update}.tar":
            reasons.append(
                f"final checkpoint update missing (expected u{expected_update}.tar.xz)"
            )
        try:
            with tarfile.open(archives[-1]) as archive:
                if not any(name.endswith("models.pt") or name.endswith("seac.pt")
                           for name in archive.getnames()):
                    reasons.append("checkpoint contains no model state")
        except tarfile.TarError as error:
            reasons.append(f"checkpoint unreadable: {error}")
    if not metadata.get("experiment", {}).get("sources"):
        reasons.append("source/commit provenance missing")
    repositories = metadata.get("experiment", {}).get("repositories", [])
    if not any(repository.get("commit") for repository in repositories):
        reasons.append("Git commit provenance missing")
    return reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="?", default=Path("seac/seac/results/sacred"))
    args = parser.parse_args()
    compatible = []
    for run in sorted((p for p in args.runs.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)):
        reasons = audit(run)
        print(f"{run.name}: {'COMPATIBLE' if not reasons else 'NOT COMPARABLE'}")
        for reason in reasons:
            print(f"  - {reason}")
        if not reasons:
            compatible.append(run.name)
    print(f"Compatible runs: {', '.join(compatible) if compatible else 'none'}")
    return 0 if compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())
