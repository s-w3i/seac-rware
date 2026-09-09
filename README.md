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

Each process writes separate logs, models, videos, and TensorBoard output under
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
