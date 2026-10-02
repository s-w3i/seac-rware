# Stage 4: fixed-map, two-method adaptation

This runner implements MAPPO (`local`) and communication + CCPD
(`communicating_ccpd`) with training seeds **0 and 1** on the adopted 40×20 physical-interval
map (41×21 RWARE positions). It reuses the Stage 3 7×7 actor, centralized critic,
learner, rewards, simulator and communication channel. It does not establish
generalization to other maps.

Each run converts its **own Stage 1 `last.pt`**. Compatible actor weights are
copied; the resized grid projection and critic are freshly initialized using
the run's training seed. Optimizers start fresh. Existing Stage 1 and Stage 3
checkpoints/results are not modified. The roughly 20M pretraining steps are
recorded separately from new adaptation steps.

| Robots | Updates | Joint steps |
|---|---:|---:|
| 1 | 98 | 200,704 |
| 10 | 98 | 200,704 |
| 20 | 196 | 401,408 |
| 30 | 196 | 401,408 |
| 40 | 683 | 1,398,784 |
| 50 | 683 | 1,398,784 |
| Total per run | 1,954 | 4,001,792 |

All four runs total **16,007,168 new joint steps** (592,674,816 agent
transitions), excluding disposable profiles and evaluation. Production retains
eight environments, 256 rollout steps, and the existing PPO settings.
Stage 4 pins deterministic float32 operations and disables TF32: the 50-robot
A2000 smoke test found roughly `8e-4` PPO replay drift with cuDNN TF32 enabled,
versus below `3e-6` with it disabled. The replay threshold remains `2e-5`.

## Launch

From `/home/utar/seac-rware`, after the smoke tests:

```bash
nohup .venv/bin/python -u stage3_runtime/scripts/run_stage4.py \
  --campaign --jobs 2 --devices cuda:0 cuda:1 \
  --output results/stage4_40x20_7x7 \
  > stage4_campaign.log 2>&1 &
```

This first runs **two disposable complete rollout/update cycles for every method
at 20, 40 and 50 robots on each listed GPU**, plus a 500-step CPU evaluation timing.
It writes memory/timing results and `cost_estimate.json`. Only after every profile
passes does the campaign start training, one process per GPU. MAPPO uses GPU 0
and communication + CCPD uses GPU 1. Seed 0 runs first for both methods; after
both finish, seed 1 runs with the same GPU assignments. It then locks all final checkpoints, evaluates the complete held-out
suite and writes `STAGE4_REPORT.md` and `summary.json`.

The GPUs must retain 20% memory headroom, with at least 2 GiB available host memory
during profiling. No automatic CPU fallback or batch-size change is permitted.
`--jobs 2` also runs the actual pair concurrently at 50 robots, one method per GPU,
and requires the same memory headroom and no more than 25% per-job slowdown.
Otherwise the campaign stops before training; `--jobs 1` is an explicit serial alternative.
Two models on the same 6 GB GPU are unsupported: the full-size local + combined
capacity check exceeded the 80% memory-use limit. Do not combine both GPU assignments.

To explicitly bypass startup profiling, add `--skip-preflight` to the campaign
command. This resumes saved checkpoints directly on the two GPUs and records
the override in `preflight_overrides.jsonl`. It skips both individual and joint
memory/timing profiles; checkpoint provenance, PPO replay, finite-value checks,
scheduled validation and fixed training budgets remain enforced.

Rerun the same command to resume. Completed, verified stages and profiles are
reused. Watch `stage4_campaign.log`, `status.json`, per-run `logs/`,
`failure.json` and `evaluation_status.json`. Progress failures never shorten a
run or grant extra training. Technical errors stop the campaign and remain in
its record even after recovery.

## Individual operations

```bash
# Convert and verify all four initial checkpoints without training.
.venv/bin/python stage3_runtime/scripts/run_stage4.py --prepare

# GPU preflight only, without starting the study.
.venv/bin/python -u stage3_runtime/scripts/run_stage4.py \
  --profile-all --devices cuda:0 cuda:1

# One complete curriculum after its GPU preflight; explicit restart permission.
.venv/bin/python -u stage3_runtime/scripts/run_stage4.py \
  --train --condition local --seed 0 --device cuda:0 --resume

# Read completed evidence and refresh the report; never runs evaluation.
.venv/bin/python stage3_runtime/scripts/run_stage4.py --report

# These require every training stage to be complete and verified.
.venv/bin/python stage3_runtime/scripts/run_stage4.py --lock-evaluation
.venv/bin/python -u stage3_runtime/scripts/run_stage4.py --evaluate
```

`--output` is supported for every operation. `--source-checkpoint` can override
one explicitly named condition/seed before the protocol is frozen. Identity and
hash checks still apply. `--agents` selects a stage for `--train` or `--profile`;
training a later stage requires its predecessor's completed artifacts.
`--schedule` accepts increasing `fleet:updates` pairs. Production requires the
table above; custom schedules require `--smoke` in a separate output directory.

Protocols freeze source hashes, map, source checkpoints, seed sets, budgets,
analysis rules and per-stage configuration. A changed protocol is rejected;
use a new output directory for a new experiment. A resumed stage restores
model/optimizer/global RNG state but **restarts its environments, recurrent
memory and independent action streams** with a recorded restart-specific seed.
It is reproducible from that restart boundary, not bitwise equivalent to an
uninterrupted rollout. Completed-update accounting is preserved.

### Runtime correction for PPO replay drift

The 40-robot MAPPO seed-0 run stopped after saved update 455 when replay error
reached `2.288818359375e-5`, above the unchanged `2e-5` limit. CUDA actor forwards
now use fixed 64-row blocks, padding and discarding unused rows. This gives
collection and recurrent replay identical arithmetic shapes. PPO minibatches,
eight environments, 256-step rollouts, float32 parameters, architecture and
training allocations remain unchanged. The additional kernel calls can increase
update time; disposable verification records the measured costs.

An existing study may adopt this verified correction through a
`runtime_repair.json` receipt that pins its original protocol hash and exact
before/after source hashes. The original protocol, checkpoints and completed
update counters are not rewritten. Any other source change is still rejected.
New checkpoints, restart records and completed-stage summaries identify their
runtime source hash, and the final report includes the repair record.

While communication + CCPD seed 0 is still running under the original campaign,
resume only the failed MAPPO worker in another terminal:

```bash
.venv/bin/python -u stage3_runtime/scripts/run_stage4.py \
  --train --condition local --seed 0 --device cuda:0 --resume \
  --output results/stage4_40x20_7x7
```

After both seed-0 workers finish and the old campaign exits, rerun the campaign
command above to continue seed 1. The old campaign can still report the original
worker failure; the resumed worker records recovery separately. Completed stages
are verified and reused. Do not launch a second campaign while the first holds
the pipeline lock.

## Validation and checkpoint selection

Validation runs at initialization, every 49 updates, and each stage endpoint:
ten scenarios (100000–100009), three action replicates, 5,000 uninterrupted steps.
Every robot's peak task age is retained, including the final age of completed
cycles, so later recovery cannot hide an earlier stall.

A checkpoint passes the operational progress screen only when **every validation
episode** has fleet completion gaps below 500 steps and every robot's task age
stays below 500. This is not a physical safety or deadlock certification.
Among passing checkpoints, select highest throughput, then lowest p95 unfinished
age, then earliest update. If none passes, minimize the fraction of episodes
failing either progress criterion, then use the same tie-breakers. Initialization
is an eligible checkpoint. Always complete the full allocation.

At fleet changes, transfer the selected actor and critic and reset optimizers,
environment state and recurrent memory. Export both selected and final actors.
Separately evaluate the selected one-robot actor on 100 validation scenarios
(100100–100199), three action replicates: stop at its first completed cycle or
500 steps. Report the empirical 99% completion target. Unfinished tasks count
as failures. This diagnostic never changes training budgets.

## Held-out evaluation and analysis

Production seeds **300000–300049**, changed-start/task variants and derived
channel/action streams are reserved. The environment rejects them by default.
The final evaluator explicitly unlocks access only after verifying all training
summaries and freezing selected/final checkpoint hashes plus analysis rules.
Seed 200000 is not reused as fresh holdout evidence.

At 40 and 50 robots, both methods and both training seeds receive:

- Selected checkpoint: 50 scenarios × three replicates × 10,000 continuous steps.
- Final checkpoint: first ten scenarios × one replicate × 10,000 steps.
- Selected checkpoint, first twenty scenarios × one replicate: deterministic
  execution, changed starts and changed tasks.
- Communicating methods, same first twenty scenarios × one replicate: packet
  loss 0%, 30%, 100%, and delay two with 10% loss. Primary evaluation supplies
  the paired normal-channel reference.

Files carry exact identities and completion receipts containing input/output
hashes. Evaluation resumes an interrupted file at its next uncompleted episode.
Complete files with inconsistent hashes or labels are rejected. Reports do not
issue final comparisons while required evidence is missing.

Reports compare communication + CCPD versus MAPPO at both
target fleets. Paired bootstrap sampling clusters by training seed and scenario,
keeping action replicates inside scenario blocks (10,000 draws, seed 1729).
Only two training seeds limit uncertainty estimates. This comparison cannot
separate the effects of communication and CCPD. No failures observed is
not proof of zero risk, and equal task seeds match initial randomness rather
than the entire policy-dependent stream of later jobs.

Recommend added complexity only at both target fleets with either at least 3%
throughput gain and a positive paired 95% interval, or a supported reduction in
progress failures with the throughput interval above the −2% non-inferiority
margin. Per-seed safeguards require no observed candidate progress failure,
throughput within 2% of the comparator, and no worse robot-failure fraction,
aged-robot time, p95 unfinished age or maximum task age. Unresolved technical
failures prevent promotion. Otherwise retain the simpler reference provisionally.

Completed-cycle p95/p99 statistics exclude unfinished cycles, whose endpoint and
maximum ages are reported separately. Raw per-robot service, stationary/repetition,
denial reasons and communication counters remain available in episode files.

## Smoke and regression checks

```bash
PYTHONPATH=stage3_runtime/seac/seac:stage3_runtime/robotic-warehouse \
  .venv/bin/python -m pytest -q \
  stage3_runtime/seac/tests/test_stage4.py \
  stage3_runtime/seac/tests/test_stage3.py \
  stage3_runtime/seac/tests/test_navigation.py \
  stage3_runtime/seac/tests/test_shared_ppo.py

.venv/bin/python -u stage3_runtime/scripts/run_stage4.py \
  --train --smoke --schedule 1:2,10:2 --condition communicating_ccpd \
  --seed 0 --device cuda:0 --output results/stage4_smoke/manual
```

Smoke mode uses two environments, 32-step rollouts, short evaluations and a
separate immutable protocol. Smoke final evaluation uses **100500–100501**,
never the production holdout. It cannot satisfy production profile gates or
produce research recommendations. Smoke training means only disposable small
updates; no production campaign is launched by these checks.
