# Training and provenance

## What is reproducible

1. Load the exact released trained actors, with no training or downloads.
2. Recompute the published tables from original, checksummed episode evidence.
3. Re-run original validation scenarios with the supplied simulator/policy and pinned software.
4. Run a new adaptation curriculum from the supplied original Stage-1 sources.

The original run was interrupted and resumed with environment/recurrent-state restarts and a numerical repair. A new uninterrupted training run is **not guaranteed to reproduce identical weights**. The original protocols, recorded runtime hashes, repair receipt and completed-stage summaries are preserved as historical evidence, including their original absolute paths. They must not be rewritten to suggest a fresh run matches the original provenance. Source-path/config changes generate a new protocol in a new output directory.

## Completed experiment

Two methods, training seed **0** only. The original plan included seed 1, but it was cancelled after seed 0. Stage-1 seed-1 source checkpoints are included solely to make the original conversion/regression tests self-contained; there are no Stage-4 seed-1 results. No independent holdout evaluation was performed.

| Robots | Updates | New joint steps |
|---|---:|---:|
| 1 | 98 | 200,704 |
| 10 | 98 | 200,704 |
| 20 | 196 | 401,408 |
| 30 | 196 | 401,408 |
| 40 | 683 | 1,398,784 |
| 50 | 683 | 1,398,784 |
| Total | 1,954 | 4,001,792 |

Eight environments, 256 rollout steps; recurrent actor; pooled centralized critic. Compatible Stage-1 actor weights transfer, the resized grid projection and critic initialize freshly, and optimizers reset. At fleet transitions, the selected actor/critic transfers and optimizers/environment/recurrent states reset. Complete every allocation, regardless of learning performance.

Validation: initialization, every 49 updates, and fleet endpoint; seeds 100000–100009, three action replicates, 5,000 continuous steps. Select highest throughput among checkpoints passing the progress screen (no fleet gap or any robot task age ≥500), then lower p95 unfinished age, then earlier update. If none passes, minimize failure fraction before the throughput tie-breakers. This is an operational screen, not proof of deadlock or safety.

## Smoke only

```bash
python scripts/train_seed0.py --train --smoke --condition local --seed 0 \
  --device cpu --schedule 1:2,50:2 --output results/smoke_local
python scripts/train_seed0.py --train --smoke --condition communicating_ccpd --seed 0 \
  --device cpu --schedule 1:2,50:2 --output results/smoke_combined
```

These disposable small batches verify plumbing; they do not establish full-batch GPU memory capacity.

## Production adaptation (explicit opt-in)

The original hardware was two 6 GB RTX A2000 GPUs. Use one model per GPU, and keep 20% GPU headroom and 2 GiB available host RAM. Do not combine both jobs on one 6 GB GPU. Startup profiling, replay-error checks (≤2e-5), finite-value checks, and scheduled validation remain enabled.

First prepare the shared immutable protocol once:

```bash
python scripts/train_seed0.py --prepare --output results/stage4_reproduction
```

Run the disposable full-batch profiles and joint memory/runtime check before launching training:

```bash
python -u scripts/train_seed0.py --profile-all --devices cuda:0 cuda:1 \
  --output results/stage4_reproduction
python -u scripts/train_seed0.py --profile-pair --devices cuda:0 cuda:1 \
  --output results/stage4_reproduction
```

If either check fails, do not launch concurrent training. After both checks pass, run these in two terminals:

```bash
# Terminal 1
python -u scripts/train_seed0.py --train --condition local --seed 0 \
  --device cuda:0 --resume --output results/stage4_reproduction
```

```bash
# Terminal 2
python -u scripts/train_seed0.py --train --condition communicating_ccpd --seed 0 \
  --device cuda:1 --resume --output results/stage4_reproduction
```

Each exits after its seed-0 curriculum and endpoint validation. No seed-1 job or holdout evaluation starts automatically. Resume with the same command; the runner records environment/recurrent restarts rather than pretending to restore an unrecorded simulator state. The historical `--skip-preflight` option exists but is not recommended on unprofiled hardware.

## Numerical repair and portability

The original MAPPO n40 run stopped after saved update 455 at replay error 2.288818359375e-5. TF32 was already disabled. Fixed 64-row CUDA actor-forward blocks removed batch-shape-dependent numerical drift; the threshold was not relaxed. Both methods later resumed under the repaired runtime. This implementation is included unchanged.

Release runtime Python files are byte-identical to the final working runtime. Only the map manifest's absolute path was changed to a relative bundled map; the geometry hash remains 11d6084e865626e525e6b6e391a206ebc105e3abded677344bac4e0739613c5e. Release wrappers and tests live outside that frozen runtime. Original manifest and before/after source hashes are in `evidence/`. The hash manifest is an integrity check, not an external digital signature.

The historical full-campaign evaluator still expects two completed training seeds. It cannot honestly label this one-seed release a completed original campaign. Reserved seeds 300000–300049 and derived streams remain guarded. A future one-seed holdout study needs its own locked protocol; this release does not silently unlock them.
