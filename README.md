# SEAC + RWARE custom five-robot baseline

This workspace contains clean copies of the original SEAC and RWARE repositories,
with the custom 10x6 warehouse registered as `rware-custom-5ag-v2` and a
matching SEAC training config added.

The custom environment now enables a seeded `TaskManager`. Every robot receives
a unique active rack task and observes its phase plus relative target position.
Its policy action space contains only `NOOP`, `FORWARD`, `LEFT`, and `RIGHT`;
pickup, delivery, unloading, and the next assignment are automatic.

Every environment step reports task and navigation measurements in `info`.
Robot-level values are arrays ordered by robot: `completed_cycles`,
`deliveries`, `pickup_time`, `delivery_time`, `return_time`, `path_length`,
`wait_steps`, `conflict_attempts`, and `movement_denied`. Timing is emitted on
the step that completes each phase; other entries are per-step events.
`deadlock_events` is a team event emitted once after ten consecutive steps with
at least one blocked forward request and no position or orientation change.

Source revisions:

- SEAC: `29dcbce5cb82d05361b22a2815dd5d0e54fd83b5`
- RWARE: `43018983b5e42cd8050481a871eca0baf556ac7e`

## Setup

The compatibility layer uses current Gymnasium and a small native synchronous
vector runner. PyTorch 2.5.1 CUDA 12.1 is the newest CUDA build compatible with
this machine's NVIDIA 535 driver. Create an environment and install RWARE plus
the SEAC requirements:

```bash
cd /home/usern/seac-rware-custom-5
python3 -m venv .venv
.venv/bin/pip install -r seac/requirements.txt
.venv/bin/pip install -e robotic-warehouse
```

## Train

```bash
cd /home/usern/seac-rware-custom-5/seac/seac
../../.venv/bin/python train.py with rware_custom_5ag
```

Run the short GPU smoke training (40 environment steps) with:

```bash
cd /home/usern/seac-rware-custom-5/seac/seac
../../.venv/bin/python train.py with rware_custom_5ag_smoke
```

To force CPU for troubleshooting, append `algorithm.device=cpu` to either
training command.

The standalone map copy is in `assets/warehouse-10-6.map`. RWARE's registered
environment embeds the equivalent layout directly so it works from any current
working directory.

## Phase 2 routing mode (implemented; learning not evaluated)

`rware-custom-5ag-routing-v2` uses the same 10-row × 6-column map, 21 racks,
workstation, seeded task manager, and five robots. It enables
`routing_features_enabled=True` and 5×5 sensing. The original
`rware-custom-5ag-v2` registration and baseline behavior are preserved.
The matching SEAC named configuration is `rware_custom_5ag_routing`.
No training was run as part of Phase 2 verification.

Routing mode supports flattened and dictionary observations with individual
rewards and no messages. Its default flattened observation has 199 values per
robot, versus 79 for the original custom baseline. Baseline checkpoints cannot
be loaded into this larger input layer; start a fresh routing-mode model.
Image observations are not supported in this phase.

The task observation retains its six visible navigation/service phases and
normalized target displacement, then appends distance (1 value), previous
navigation action (4), movement success (1), blocked streak (1), and previous
distance (1). Automatic service records NOOP; collision-rejected forward
attempts still record FORWARD, with movement success false. History resets
between episodes. The blocked streak counts only consecutive robot-rejected
forward attempts, saturating at 10 in the observation. Previous distance resets
to the new distance when the target or load state changes.

Distance is a four-neighbor BFS translation count divided by map cell count.
It ignores robots and orientation. Unloaded robots can traverse rack cells;
loaded robots use reset-time rack locations as obstacles, except their own
rack's home cell and current cell. This fixed reference does not exploit
other racks' temporary vacancies. An unreachable target has normalized distance
1 and contributes no progress shaping.

The per-robot reward is:

```text
progress_weight * (distance_before - distance_after)
- step_cost - conflict_cost * robot_blocked - stall_cost * prolonged_block
+ event_bonus
```

Environment keyword defaults are `progress_weight=0.1`, `step_cost=0.01`,
`conflict_cost=0.1`, `stall_cost=0.2`, `pickup_reward=1.0`,
`delivery_reward=2.0`, and `return_reward=3.0`. Progress uses unnormalized
translation counts against the same pre-step target/load state. Service steps
have no progress reward, but still incur the step cost. Event bonuses replace
baseline delivery rewards and occur once per completed automatic event.
The stall cost applies on the tenth and every subsequent consecutive
robot-blocked forward attempt. Boundary and standing-rack rejection incur no
conflict/stall cost. This is heuristic shaping, without a policy-invariance claim.

All original step metric names remain available. In routing mode:

- `wait_steps` counts navigation NOOP and denied forward requests, excluding
  turns and service steps; `movement_denied` includes all rejected forward requests.
- `conflict_attempts` counts attempted vertex/edge conflicts, while `robot_blocked`
  counts physically valid forward requests actually rejected by robot resolution.
- `deadlock_events` is a team-stall proxy, not a proof of deadlock: it fires once
  after ten steps with robot rejection, no translation/rotation, and no phase
  progress, then rearms when that condition breaks.
- `pickups` counts pickup events; `cycle_time` emits elapsed steps only on a
  completed return. Phase and cycle durations include automatic service steps.
- `reward_progress`, `reward_step`, `reward_conflict`, `reward_stall`, and
  `reward_event` are per-robot arrays that sum to the emitted reward.

`RecordEpisodeStatistics` emits `info['episode_metrics']` on termination or
truncation. It contains team event totals, completion-based phase/cycle means,
cycles per 1,000 joint environment steps, and unfinished-cycle count (tasks
that have consumed at least one step but have not completed). Undefined means
are omitted. Training/evaluation aggregate completed episodes only, weight
phase/cycle means by completion counts, and calculate throughput from total
cycles divided by total episode steps. Wall-clock FPS remains a separate measure.
Statistics reset with the episode; unfinished cycles do not enter duration means.

To launch Phase 0 and Phase 2 as separate training processes on a host machine:

```bash
./scripts/run_phase0_phase2.sh
```

Phase 0 uses the original feedforward SEAC on `rware-custom-5ag-v2` with
routing features disabled; Phase 2 uses the routing-enabled variant on the
same map. Each process writes separate logs, models, videos, and TensorBoard output under
`results/parallel-<timestamp>`. Both default to `cuda:0`; on a host with two
GPUs, use `PHASE0_DEVICE=cuda:0 PHASE2_DEVICE=cuda:1`. Set `PYTHON_BIN` or
`RUN_DIR` to override the Python executable or output directory. The script
waits for both processes and exits nonzero if either run fails.

Verification from the workspace root, without changing package installations:

```bash
PYTHONPATH="$PWD/robotic-warehouse" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest robotic-warehouse/tests -q
```

Plugin autoload is disabled to avoid the host ROS pytest plugin's missing `lark`
dependency. All **73 tests passed**; the one existing warning concerns standard
RWARE's multi-agent reward list versus Gymnasium's scalar-reward checker.
Tests include a scripted five-robot warehouse cycle, baseline regression,
observation-space checks, reward and collision cases, episode accounting, and
an existing SEAC policy forward pass. No optimizer updates are performed.

## PPO, PPO-GRU, MAPPO and MAPPO-GRU

The separate shared PPO pipeline supports all four variants. Existing SEAC/A2C
entry points and configurations remain available. The only simulator behavior
change is that an internal elapsed-step limit now reports **truncation**, while
inactivity still reports termination. The legacy runner already handles both.

| Named configuration | Actor | Critic |
|---|---|---|
| `shared_ppo_routing` | Shared MLP | Shared local MLP |
| `shared_ppo_gru_routing` | Shared MLP + GRU | Shared local MLP + GRU |
| `mappo_routing` | Shared MLP | Centralized team MLP |
| `mappo_gru_routing` | Shared MLP + GRU | Centralized team MLP |

Every robot uses the same actor weights and its own GRU memory when enabled.
Actor inputs remain local. Both algorithm families optimize mean fleet reward;
MAPPO changes critic information, with one team value target per transition.
Actor/critic parameters and Adam optimizers are separate. Automatic-service
steps advance memory and train values but have no actor/entropy/KL loss.
Truncations bootstrap from final observations, before reset.

Defaults are eight synchronous environments, 256-step rollouts, four PPO epochs,
four minibatches, actor/critic LR `3e-4`, gamma `0.99`, GAE lambda `0.95`, PPO
clip `0.2`, entropy `0.01`, gradient clipping `0.5` and target KL `0.02`. GRU
updates use 32-step sequences and up to 16 burn-in steps, retaining history
across rollouts. Stored recurrent states remain an approximation after policy
updates; burn-in reduces this mismatch. KL stopping affects the actor only.
No dependency upgrades, mixed precision or distributed model training are used.

### Standalone training

Run from the repository root; substitute any configuration from the table:

```bash
.venv/bin/python seac/seac/train_shared.py with shared_ppo_routing \
  device=cpu num_env_steps=8192 run_dir=results/shared_manual/ppo_smoke

CUDA_VISIBLE_DEVICES=1 .venv/bin/python seac/seac/train_shared.py with mappo_gru_routing \
  device=cuda:0 num_env_steps=2000000 run_dir=results/shared_manual/mappo_gru_seed0
```

`run_dir` must not exist; omission generates a unique directory. Configuration
files resolve relative to the trainer, so the entry point works from other
working directories too. Environment-step budgets count joint warehouse
transitions, not robot actions. Runs finish complete rollouts and report any
budget overshoot. The default training horizon is 500 steps with one time-limit
owner. `num_envs` is synchronous batching, not a count of CPU workers.

### Two-GPU launcher

```bash
# Inspect commands and GPU assignments without creating files or requiring CUDA.
.venv/bin/python scripts/run_shared_baselines.py --gpus 0 1 --seeds 0 \
  --num-env-steps 8192 --attempt smoke --dry-run

# Run all four models in two waves, with one child per GPU.
.venv/bin/python scripts/run_shared_baselines.py --gpus 0 1 --seeds 0 \
  --num-env-steps 8192 --attempt smoke

# Matched training seeds; each seed finishes before the next starts.
.venv/bin/python scripts/run_shared_baselines.py --gpus 0 1 --seeds 0 1 2 \
  --num-env-steps 2000000 --attempt ff_and_gru

# Optional capacity experiment: all four children simultaneously. Not benchmarked yet.
.venv/bin/python scripts/run_shared_baselines.py --gpus 0 1 --seeds 0 \
  --num-env-steps 8192 --jobs-per-gpu 2 --attempt four_at_once
```

GPU 0 runs PPO then PPO-GRU; GPU 1 runs MAPPO then MAPPO-GRU. With
`--jobs-per-gpu 2`, each GPU runs both of its models together. Use `--models`
with named configurations to select a subset. `--num-envs`, `--rollout-steps`,
`--env-name`, `--eval-steps`, `--python`, and `--output` provide explicit overrides.
GPU indices refer to physical indices reported by `nvidia-smi`; the launcher
resolves UUIDs, isolates each child and passes logical `device=cuda:0`.
A conflicting inherited `CUDA_VISIBLE_DEVICES` restriction is rejected.

Occupied GPUs are rejected by default. Append `--allow-busy` only when you intend
to share with an existing process; the smoke validation used this because an
older SEAC-GRU run was active on GPU 0. Existing jobs are never stopped. Child
failure stops its sibling(s), and interruption terminates/reaps launcher-owned
children. Existing attempt directories are rejected. Dry-run does no GPU probing;
actual execution validates both GPUs before launching training.

Launcher outputs live under
`<output>/<configuration>/seed_<seed>/<attempt>/`, with `launch.json`, `train.log`
and a `train/` subdirectory for training artifacts. Each child has independent
Sacred and TensorBoard directories, checkpoints and metrics.

Each model's `train.log` progress line also prints `team_reward_mean` (the
mean of fleet-mean rewards across the rollout's environment steps),
`reward_sum_per_robot` (five rollout totals, in robot order), and
`reward_components_sum` (each component summed across all robots and steps).
These are rollout statistics, not average episode returns. `team_reward_mean`
is also recorded in JSON metrics and TensorBoard. The top-level launcher log
contains launcher messages; model progress is in the individual `train.log` files.

### Checkpoints, resume and evaluation

Each training directory contains `last.pt`, `best.pt`, an actor-only `actor.pt`,
`metrics.jsonl`, raw `evaluation.jsonl`, `summary.json`, `provenance.json`, source
snapshots and `source.diff`. Provenance records the actual registered map hash
and layout separately from the reference asset hash. Metrics include losses,
KL, clipping, entropy, gradients, task/reward totals, timing and peak GPU
allocation. Validation defaults to seeds 1000–1003 for 500 steps every 250,000
training steps and at the final rollout. `best.pt` maximizes validation cycles
per 1,000 steps; ties retain the earlier checkpoint. Undefined completed-cycle
duration statistics are `null`, not zero.

```bash
# Extend a run; use the original named config and a new output directory.
.venv/bin/python seac/seac/train_shared.py with shared_ppo_routing \
  device=cpu num_env_steps=16384 \
  resume=results/shared_manual/ppo_smoke/last.pt \
  run_dir=results/shared_manual/ppo_resumed

# Actor-only matched episode evaluation; defaults to test seeds 2000–2049.
.venv/bin/python seac/seac/evaluate_shared.py \
  results/shared_manual/ppo_smoke/actor.pt \
  --output results/shared_manual/ppo_test.jsonl

# Continuous 10,000-step evaluation, with no internal/outer episode cutoff.
.venv/bin/python seac/seac/evaluate_shared.py \
  results/shared_manual/ppo_smoke/actor.pt --continuous --seeds 2000 \
  --output results/shared_manual/ppo_continuous.jsonl
```

Resume restores model, optimizer, RNG and global counters, but starts fresh
environments and GRU memories using `seed + saved_env_steps`. It is not exact
trajectory continuation. Only device, output, budget and save/evaluation interval
overrides may differ. Historical best weights are copied from a matching sibling
`best.pt`; if unavailable, best selection restarts and provenance records that.
Evaluation loads either the actor export or a full checkpoint without constructing
a critic. Default actions are deterministic; `--stochastic` uses an isolated RNG
stream. Reports include raw cycle durations, p95, unfinished task ages, per-robot
rewards/cycles, denied-movement rates and inference latency. Output files must be new.

### Validation results

Validated with two RTX A2000 GPUs (6,138 MiB each), PyTorch 2.5.1+cu121 and the
existing virtual environment. Four CPU runs completed 4,096 steps each, and all
four resumed successfully to 6,144 total steps. Two GPU waves completed 8,192
steps per model, with four optimizer updates and verified concurrent child
lifetimes. All runs had finite diagnostics and changed actor/critic weights;
reloaded GPU exports matched full-checkpoint actor outputs.

| Model | GPU | GPU run seconds | Peak PyTorch allocation (MiB) |
|---|---:|---:|---:|
| PPO | 0 | 24.06 | 126.52 |
| MAPPO | 1 | 22.34 | 115.12 |
| PPO-GRU | 0 | 33.74 | 141.91 |
| MAPPO-GRU | 1 | 27.06 | 125.49 |

Timing includes final validation, excludes process startup, and was measured
while the pre-existing GPU 0 job and some CPU checks were active. Allocation
excludes CUDA context and non-PyTorch memory. These numbers do not establish
four-job throughput or learned routing quality: validation throughput remained
zero at these short budgets. A separate MAPPO-GRU actor-only 10,000-step continuous
run completed without a 500-step reset. Long training and four-at-once capacity
assessment remain deferred.

All **106 tests passed**, with the existing Gymnasium multi-agent reward warning.
Run the algorithm, launcher, legacy comparison and simulator regressions with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  seac/tests/test_shared_ppo.py seac/tests/test_shared_launcher.py \
  seac/tests/test_comparison.py robotic-warehouse/tests -q
```

## CCPD v0: current-rollout successful-event learning

`mappo_gru_ccpd_routing` adds a training-only auxiliary action loss to the
existing MAPPO-GRU. The actor, centralized critic, 199-feature observations,
reward, PPO update count and actor export format are unchanged. No replay buffer,
new dependency or deployment-time event detector is used. The critic and runner
remain specific to the current five-robot benchmark.

Tracing is opt-in through `coordination_trace_enabled` on the simulator and
`coordination_trace` on `SharedEnvs`. Records contain pre-step task/load/position,
target, requested action, decision eligibility, local density, unsaturated
blocking history, actual movement/conflict outcomes and raw distance changes.
Distances before and after always use the same pre-step target/load context;
automatic service produces zero navigation progress. The collector retains the
sampled policy action even when service executes NOOP, and captures traces before
episode reset. Neither observations nor the simulator RNG is changed.

Events start on actual robot blocking. Resolution requires three conflict-free
steps with at least one translation, within a 32-step horizon, then eight further
confirmation steps. Confirmation must show net progress of at least one cell,
valid distances, no conflict in its final three steps, and positive quality:

```text
Q = min(post-resolution progress, 4)
    - 0.1 * resolution duration
    - 0.5 * additional pre-resolution conflict steps
    - 1.0 * confirmation conflict steps
```

Events cannot overlap for one robot or cross episode/rollout boundaries. Missing
follow-up is censored; timed-out events are failed. Task changes do not produce
distance jumps because progress sums only correctly paired per-step differences.
These are v0 heuristics: the actor can succeed for reasons unrelated to its own
actions, and the positive-advantage filter uses the existing **team** advantage.

Only actions through provisional resolution are teaching candidates; confirmation
steps label the outcome. Automatic service, denied forward requests and nonpositive
**raw, pre-normalization** advantages are excluded. Selected actions reuse their
existing GRU sequence, burn-in and reset context. Within each actor minibatch:

```text
actor loss = existing PPO actor loss + ccpd_coef * mean(-weight * log_prob)
```

Scores are divided by the selected-score mean and capped at 2. No critic loss,
optimizer pass or reward is added. Empty selections use ordinary PPO. The existing
actor KL stopping and gradient clipping apply to the combined update.

| Configuration key | Default |
|---|---:|
| `ccpd_mode` | `off`; CCPD named config uses `successful` |
| `ccpd_coef` | 0.01 |
| `ccpd_max_sample_fraction` | 0.10 of eligible rollout decisions |
| `ccpd_max_noop_fraction` | 0.25 of selected actions |
| `ccpd_event_horizon` / `ccpd_clear_steps` / `ccpd_confirmation_steps` | 32 / 3 / 8 |
| `ccpd_progress_cap` / `ccpd_progress_weight` | 4 / 1 |
| `ccpd_duration_cost` / `ccpd_conflict_cost` / `ccpd_recurrence_cost` | 0.1 / 0.5 / 1 |
| `ccpd_min_progress` / `ccpd_min_quality` | 1 / 0 |
| `ccpd_trace_events` | false |

Modes are `off`, `successful`, `all_conflict` and `random`. The latter two use
fully observed conflict events or arbitrary eligible transitions respectively,
with the same action restrictions. Each rollout's successful pool determines
the feasible sample count and weight multiset for all three active modes.
Sampling is without replacement and respects the NOOP cap. Controls randomly
assign that weight multiset to their selected actions. Counts are comparable by
rule, not identical across policies whose trajectories diverge. If there are no
eligible successful actions, all three active modes skip the auxiliary update.
Sampling has a separate generator derived from seed/update and does not advance
the policy RNG.

`ccpd_mode=off` or `ccpd_coef=0` bypasses CCPD collection/selection/loss. Old
checkpoints missing CCPD settings resume with disabled defaults; active modes
cannot be enabled by silently resuming a plain baseline. CCPD settings remain
fixed during resume. The saved update count restores selection RNG indexing;
environment and GRU state still restart according to the existing resume policy.

Metrics include outcomes/rejection reasons, quality, resolution duration,
recurrence, raw-advantage statistics, selected/NOOP fractions, auxiliary loss,
main actor loss, and the weighted auxiliary/main magnitude ratio. A nearly zero
main loss can make that ratio large; it is diagnostic, not an automatic weight
controller. With `ccpd_trace_events=True`, `coordination_events.jsonl` contains
up to 20 event traces per rollout for the first four updates. Provenance records
the detector version, full resolved configuration and simulator/module snapshots.

### Commands and later comparison protocol

Run from `/home/utar/seac-rware`. Commands create new output directories and
refuse to overwrite existing runs.

To train all three CCPD seeds **at the same time**, with seed 0 on physical GPU 0
and seeds 1 and 2 sharing physical GPU 1, run:

```bash
/home/utar/seac-rware/.venv/bin/python /home/utar/seac-rware/scripts/run_ccpd_seeds.py
```

This defaults to 20M steps per seed, enables bounded event traces, and writes
separate `train.log`, checkpoints and metrics under
`results/ccpd/mappo_gru_ccpd_routing/seed_<seed>/ccpd_v0_20m/`.
Append `--dry-run` to inspect commands, or `--attempt <new-name>` for a fresh run.
The script checks both GPUs before launch; its intentional sharing of GPU 1
does not require `--allow-busy`. That flag is only for sharing with processes
already running before launch. Ctrl+C or a failed child stops the other children
owned by this launcher.

```bash
# Short standalone validation; substitute random/all_conflict/off for controls.
.venv/bin/python seac/seac/train_shared.py with mappo_gru_ccpd_routing \
  device=cpu num_env_steps=4096 ccpd_trace_events=True \
  run_dir=results/ccpd_manual/smoke

# Read-only bounded event audit of a trained baseline; no optimizer updates.
.venv/bin/python scripts/audit_ccpd_events.py \
  results/shared_baselines/mappo_gru_routing/seed_0/four_models_20m/train/best.pt \
  --output results/ccpd_manual/event_audit

# Inspect the later three-seed training commands without launching them.
.venv/bin/python scripts/run_shared_baselines.py \
  --models mappo_gru_ccpd_routing --gpus 0 1 --seeds 0 1 2 \
  --num-env-steps 20000000 --attempt ccpd_v0_20m \
  --output results/ccpd --dry-run

# Full training, to run later: same command with --dry-run removed.
# It trains from scratch, never loading the baseline used by the event audit.

# Later held-out evaluation: default seeds 2000–2049, deterministic, 500 steps.
# Repeat for seed_1/seed_2 and for both best.pt and last.pt of every method.
.venv/bin/python seac/seac/evaluate_shared.py \
  results/ccpd/mappo_gru_ccpd_routing/seed_0/ccpd_v0_20m/train/best.pt \
  --output results/ccpd/evaluation/seed0_best.jsonl

# Run every simulator/algorithm/launcher regression, including CCPD.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest seac/tests robotic-warehouse/tests -q
```

The launcher still defaults to the original four models; CCPD must be selected
explicitly and is assigned to its MAPPO GPU slot. The CCPD named config defaults
to 20M steps, but the launcher budget is always explicit (its unchanged default
is 2M). Match seeds 0–2, eight environments, 256-step rollouts, four PPO epochs,
four minibatches, sequence length 32 and burn-in 16. Preserve validation seeds
1000–1003, 500 steps and 250,000-step validation intervals. Select `best.pt` by
validation throughput; report final `last.pt` results separately. Never select
checkpoints or tune event thresholds using held-out test results.

Later, evaluate CCPD, saved MAPPO-GRU and saved Shared-PPO-GRU on that same
held-out suite. Report cycles/1k, completed-cycle durations, raw conflict proposals,
movement denial, waiting, unfinished tasks and variation across training seeds.
Run random/all-conflict controls afterward using identical budgets. No long
training or comparative performance claim is part of this implementation.

### CCPD validation results

All **130 tests passed** with the existing Gymnasium reward-list warning.
All four CPU modes completed 4,096 steps; GPU CCPD completed 8,192 steps/four
updates with finite diagnostics and 41 selected teaching actions. Its peak
PyTorch allocation was about 125.5 MiB. Actor and critic parameters changed,
and the GPU export matched the full checkpoint after CPU reload.

The read-only trained-policy audit covered 8,192 environment steps and found
343 events: 263 successful, 47 failed and 33 censored, with 407 selected actions.
Inspected examples included a turn after blocking followed by target progress,
negative progress after delivery, recurrence during confirmation and a zero-quality
resolution. The checkpoint's actor/critic parameters were unchanged. These counts
validate data flow and labels, not CCPD's performance benefit.

Disabled CCPD also matched the saved pre-CCPD learner's actor/critic parameters,
environment observations and Torch RNG **bit for bit after three updates**.
Detailed artifacts and the validation report are under `results/ccpd_validation/`.

## Frozen CCPD diagnostic protocol

The original final-checkpoint comparison used validation seeds 1000–1003;
its interpretation is corrected in `results/comparison/final_policy_comparison.md`.
The staged diagnostic runner audits the nine existing runs, evaluates final and
validation-selected best checkpoints on held-out seeds 2000–2049, tests reflected
and rotated layouts and 10,000-step continuous operation, then trains the six
matched `random`/`all_conflict` controls at 20M steps each. Seeds 3000–3049 remain
reserved. No policy architecture, reward or task-demand changes are included.

```bash
PYTHONPATH=robotic-warehouse .venv/bin/python scripts/validate_ccpd.py all
```

Do not launch a second runner while one is active; the output directory is
locked. Completed compatible evaluations are reused, source changes are rejected,
and incomplete training attempts require explicit recovery. Training uses one
process per GPU and refuses to share occupied GPUs. Detailed usage and artifacts
are documented in [the diagnostic README](results/ccpd_diagnostic/README.md).
Read `results/ccpd_diagnostic/status.json` for execution state and
[the generated comparison](results/ccpd_diagnostic/comparison.md) for current
results; interim reports explicitly identify missing experiments. The final
report is generated automatically after control evaluation.

## One-command CCPD investigation (isolated checkout)

This checkout adds `random_action_matched` and read-only failure/selection
analysis. The original experiment and its frozen sources stay under
`/home/utar/seac-rware`. Use the original virtual environment; the new launcher
explicitly resolves this checkout's simulator and supplies that path to children.
No installation or new environment generation is required.

Run from any directory:

```bash
/home/utar/seac-rware/.venv/bin/python /home/utar/seac-rware-ccpd-investigation/scripts/run_ccpd_investigation.py
```

The command waits for the original campaign to finish, verifies all prerequisite
artifacts, analyzes existing learning histories, replays up to 21 adverse cases
across five methods, audits selection on identical frozen rollouts, trains the
8-run / 80M-step pilot, and evaluates and reports the results. Only free GPUs
are used, one training process per GPU; unrelated processes are never stopped.
Keep the command running in your terminal for the multi-hour experiment.

Outputs live in `/home/utar/seac-rware/results/ccpd_investigation/`:

- `comparison.md`, `next_steps.md`, `summary.json`: final results and screening decision.
- `status.json`, `pipeline.log`: current stage and errors.
- `analysis/`: learning-curve and auxiliary-diagnostic CSVs, evidence notes, selected cases and replay summaries.
- `traces/`: compressed per-step robot timelines and verified episode summaries.
- `selection/`: identical-rollout selection audits including action, robot, phase, density, event outcome and quality.
- `training/`, `evaluation/`: separate pilot checkpoints, logs and episode results.
- `manifest.json`: frozen code, dependency versions and prerequisite artifact hashes.
- `validation/`: test results and the 4,096-step CPU smoke run.

The pilot uses fresh seeds 0 and 1 at 10M steps for off, successful, random, and
action-matched random selection. Its final checkpoints are evaluated on the
original map (50 × 500 steps), each transformed map (20 × 500 steps), and long
runs (10 × 10,000 steps). All episode seeds start at 2000; 3000–3049 remain unused.
20M and 10M comparisons are reported separately. The action-matched mode preserves
successful selection's exact per-action quotas and weight multiset on identical
data; independently trained trajectories need not have identical selections.

A recommendation to extend to a full-budget action-matched comparison requires
fewer original-map conflicts in both pilot seeds, original-map throughput at
least 98% of both random controls, and no increase in the long-run fraction of
robot endpoints with cycle age ≥500. Two seeds and this 2% screening allowance do
not establish equivalence or publication-level superiority. The launcher does
not automatically extend the pilot budget.

`--dry-run` prints the protocol without creating outputs. `--check` runs only
regressions and the new mode's CPU smoke training. Completed work is reused only
with matching inputs and integrity receipts. Incomplete training, interrupted
artifacts, changed sources or modified completed results stop with an explicit
error; they are never silently resumed, overwritten or counted as complete.
