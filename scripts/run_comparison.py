#!/usr/bin/env python3
"""Run four methods per one process per GPU, in two waves per seed."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


METHODS = (
    "rware_phase2_seac",
    "rware_phase2_seac_gru",
    "rware_phase2_shared_a2c",
    "rware_phase2_shared_a2c_gru",
)


def command(root, method, seed, gpu, output, attempt):
    result = (output / method.replace("rware_phase2_", "")
              / f"seed_{seed}" / attempt)
    return [
        str(root / ".venv/bin/python"), "train.py", "-u", "with", method,
        f"seed={seed}", f"algorithm.device=cuda:{gpu}",
        f"save_dir={result}/models/{{id}}",
        f"eval_dir={result}/eval/{{id}}",
        f"loss_dir={result}/loss/{{id}}",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs=2, default=("0", "1"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(643231229,))
    parser.add_argument(
        "--reuse-seac", action="store_true",
        help="Skip SEAC training and reuse its matching completed baseline.",
    )
    parser.add_argument(
        "--pack-gpu1", action="store_true",
        help="With --reuse-seac, run SEAC-GRU on GPU 0 and both shared models on GPU 1.",
    )
    parser.add_argument("--attempt", default="attempt_1")
    parser.add_argument("--output", type=Path, default=Path("results/comparison"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    workdir = root / "seac/seac"
    methods = METHODS[1:] if args.reuse_seac else METHODS
    if args.pack_gpu1 and not args.reuse_seac:
        parser.error("--pack-gpu1 requires --reuse-seac")
    failures = []

    for seed in args.seeds:
        planned = []
        for index, method in enumerate(methods):
            gpu = args.gpus[0] if index == 0 else args.gpus[1] \
                if args.pack_gpu1 else args.gpus[index % 2]
            cmd = command(root, method, seed, gpu, root / args.output, args.attempt)
            run_dir = (root / args.output / method.replace("rware_phase2_", "")
                       / f"seed_{seed}" / args.attempt)
            planned.append((method, gpu, cmd, run_dir))
        existing = [str(run_dir) for _, _, _, run_dir in planned if run_dir.exists()]
        if existing and not args.dry_run:
            raise FileExistsError(
                "Attempt already exists; choose a new --attempt: " + ", ".join(existing)
            )
        if args.dry_run:
            for index, (_, _, cmd, _) in enumerate(planned):
                wave = 1 if args.pack_gpu1 else index // 2 + 1
                print(f"wave={wave} " + " ".join(cmd))
            continue
        wave_size = len(planned) if args.pack_gpu1 else 2
        for wave_start in range(0, len(planned), wave_size):
            processes = []
            for method, gpu, cmd, run_dir in planned[wave_start:wave_start + wave_size]:
                record = {
                    "method": method, "seed": seed, "gpu": gpu,
                    "wave": wave_start // wave_size + 1,
                    "command": cmd, "cwd": str(workdir),
                }
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "launch.json").write_text(json.dumps(record, indent=2))
                log = (run_dir / "train.log").open("w")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["PHYSICAL_GPU_ID"] = str(gpu)
                # CUDA_VISIBLE_DEVICES remaps the selected GPU to cuda:0.
                cmd[cmd.index(f"algorithm.device=cuda:{gpu}")] = "algorithm.device=cuda:0"
                process = subprocess.Popen(
                    cmd, cwd=workdir, env=env, stdout=log,
                    stderr=subprocess.STDOUT, text=True,
                )
                record.update(
                    pid=process.pid, started_at=time.time(),
                    environment={"CUDA_VISIBLE_DEVICES": str(gpu),
                                 "PHYSICAL_GPU_ID": str(gpu)},
                )
                (run_dir / "launch.json").write_text(json.dumps(record, indent=2))
                processes.append((method, process, log, run_dir, record))

            pending = list(processes)
            while pending:
                finished = [item for item in pending if item[1].poll() is not None]
                if not finished:
                    time.sleep(.5)
                    continue
                if any(item[1].returncode for item in finished):
                    for _, process, _, _, _ in pending:
                        if process.poll() is None:
                            process.terminate()
                for item in list(pending):
                    method, process, log, run_dir, record = item
                    if process.poll() is None:
                        continue
                    status = process.wait()
                    log.close()
                    record.update(exit_status=status, finished_at=time.time())
                    (run_dir / "launch.json").write_text(json.dumps(record, indent=2))
                    if status:
                        failures.append((method, seed, status))
                    pending.remove(item)
            if failures:
                break
        if failures:
            break

    if failures:
        for method, seed, status in failures:
            print(f"FAILED: {method} seed={seed} exit={status}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
