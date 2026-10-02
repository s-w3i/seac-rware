# Decentralized warehouse navigation — Stage 4 pretrained models

This release contains the code and **seed-0 pretrained MAPPO and communication + CCPD models** used in the fixed-map Stage 4 study. The map spans 40×20 physical intervals (41×21 simulator positions); actors use 7×7 local observations and recurrent memory. Both runs completed the 1→10→20→30→40→50-agent curriculum: **4,001,792 new joint steps per model**, after 20,000,768 Stage-1 steps.

**Scope:** one adopted map, one completed training seed. Seed 1 was cancelled by the user. Reported scores are checkpoint-selection validation, not independent holdout evaluation or proof of superiority. The reserved holdout has not been evaluated.

## Install

Use Linux and Python **3.10** (original environment: 3.10.12). CPU inference works with the pinned CUDA-enabled Torch wheel; a GPU is required for production training. Create an isolated environment without system site packages:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/pretrained.py --verify --report
```

The simulator is vendored and imported directly; do not install a different `rware` package over it. Video recording additionally needs an OpenGL display, or system packages `xvfb`, `xauth`, and Mesa/OpenGL on a headless Linux machine.

## Run the pretrained models — no training required

The selected actor-only and full actor/critic checkpoints, plus final checkpoints, are included directly in `pretrained/`. Git LFS and external weight downloads are not required. Full checkpoints include optimizer/RNG state; load only trusted checkpoints.

```bash
python scripts/pretrained.py --method local --agents 50 --steps 1000 \
  --output results/demo_local.json
python scripts/pretrained.py --method communicating_ccpd --agents 50 --steps 1000 \
  --output results/demo_combined.json
```

Both use demonstration scenario **200001**, stochastic actions, and CPU inference. Change `--agents` to `40`, or use `--kind last` to expose late-training regression. `--deterministic` is an explicitly different execution mode. Demonstrations are not fresh holdout evidence.

Record matched 50-agent videos (~50 seconds each at 20 frames/s):

```bash
xvfb-run -a python scripts/pretrained.py --method local --agents 50 \
  --video results/local_50.mp4 --output results/local_50_video.json
xvfb-run -a python scripts/pretrained.py --method communicating_ccpd --agents 50 \
  --video results/combined_50.mp4 --output results/combined_50_video.json
```

## Reproduce the reported validation

`--report` recomputes the table from the checksummed episode evidence. To actually re-run a checkpoint on the original **10 scenarios × 3 action replicates × 5,000 steps**:

```bash
python scripts/pretrained.py --method local --agents 50 --validation \
  --output results/local_50_validation.json
python scripts/pretrained.py --method communicating_ccpd --agents 50 --validation \
  --output results/combined_50_validation.json
```

Repeat at `--agents 40` and/or `--kind last` as required. Validation can take approximately 75–105 minutes per checkpoint on the original machine. The original CPU/software environment is pinned; cross-platform floating-point differences can change stochastic trajectories. No claim of bitwise cross-hardware equivalence is made.

| Fleet | Method | Selected update | Cycles / 1,000 steps | Progress-failure episodes |
|---|---|---:|---:|---:|
| 40 | MAPPO | 343 | 380.77 | 0/30 |
| 40 | Communication + CCPD | 196 | 384.41 | 0/30 |
| 50 | MAPPO | 196 | 464.53 | 0/30 |
| 50 | Communication + CCPD | 49 | 473.22 | 0/30 |

The combined model's selected throughput gains are **0.96%** and **1.87%**. At 50 agents its conflicts per navigation decision are higher. Final checkpoints lose throughput relative to selected checkpoints; see [results and limitations](docs/RESULTS.md).

## New training runs

The released Stage-1 sources permit a **new adaptation run**; they are not required for inference. See [training and provenance](docs/REPRODUCING.md) for the exact curriculum, two-GPU commands, smoke checks, and historical numerical repair. Re-running training is not an exact replay of the interrupted original trajectory.

**Do not launch the historical `run_stage4.py --campaign` command:** it schedules seed 1 and holdout evaluation. The release wrapper supports the completed seed-0 scope and requires explicitly launched individual training processes.

## Repository layout

- `stage3_runtime/`: canonical Stage 3/4 runtime, simulator, and regression tests. The historical name is retained for compatibility; this is an ordinary tracked directory, not another Git worktree.
- `scripts/`: portable release entry points for inference, reporting, video, and seed-0 adaptation.
- `pretrained/`: selected/final 40- and 50-agent checkpoints, Stage-1 sources, and checksum manifest.
- `evidence/`: compressed validation/training logs, original protocols, summaries, and repair/scope decisions.
- `docs/`: experiment details and limitations.
- `tests/`: release verification tests. Earlier implementations are available in Git history, not duplicated at the root.

MAPF World is a separate project and is not included. Original upstream attribution and simulator license are retained in `stage3_runtime/robotic-warehouse/`; SEAC attribution remains in `stage3_runtime/seac/README.md`. No new blanket license for upstream components is asserted.

## Checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests stage3_runtime/seac/tests stage3_runtime/robotic-warehouse/tests
```

Historical archive/Git-history-only and CUDA-only checks may skip when their prerequisites are unavailable. Release tests compare actor-only and full checkpoint weights and action outputs, verify all checksums, and run both policies in the simulator.
