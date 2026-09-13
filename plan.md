# Implementation plan: shared PPO and MAPPO for SEAC–RWARE

Status: all four variants (PPO, PPO-GRU, MAPPO, MAPPO-GRU) and the GPU launcher are implemented and smoke-validated.
Delivery branch: `codex/ppo-mappo-readiness`.

The design below records the original specification. The subsequently approved
implementation includes four named configs and a Python launcher with
`--jobs-per-gpu 1|2`; use the [README](README.md#ppo-ppo-gru-mappo-and-mappo-gru)
for verified commands, exact defaults, checkpoint semantics and validation results.
Research-study checklist items below remain future experiments, not claims of
completed training. Four-job capacity benchmarking and long training are deferred.
Prepared: 2026-09-13.
Repository baseline: [s-w3i/seac-rware, commit 56238fc](https://github.com/s-w3i/seac-rware/tree/56238fc8b699e257effaa48a8638f153faec69c0).

## 1. Goal and boundaries

Implement only the following two parameter-shared baselines on the existing routing simulator:

1. **Shared PPO:** independent/local value estimation, one actor shared across every robot. This is a parameter-shared IPPO baseline with a cooperative reward.
2. **MAPPO:** centralized value estimation during training, the same shared actor and decentralized execution.

MAPPO also uses the PPO objective. The main comparison changes critic information, not the policy optimizer, actor observations, reward, or motion rules. Shared PPO is the local-critic control; MAPPO is the CTDE backbone candidate. Both collect fleet experience centrally and deploy the actor independently on each robot.

Keep the existing warehouse, automatic task service, task assignment, rotations, collision resolver, routing observations, and reward components. Preserve the legacy SEAC entry point and existing research plans. Do not implement distillation, simulator branching, communication, maneuver options, new scheduling, or additional reward shaping in this milestone.

Success means a trustworthy implementation and a reproducible comparison. Neither a MAPPO win nor a SOTA result is assumed.

## 2. Repository findings that determine the implementation

| Existing component | Finding | Required response |
|---|---|---|
| `seac/seac/train.py`, `teams.py`, `shared_a2c.py` | Current trainer selects independent SEAC or already-implemented Shared-A2C; both also support GRU | Preserve those paths. Add a separate PPO/MAPPO entry point with a common clipped updater and separate actor/critic networks; Shared-A2C is not PPO |
| `seac/seac/envs.py` | Synchronous vector runner collapses termination/truncation and replaces final observations with reset observations | Add an explicit transition API that preserves final observations and critic inputs |
| `seac/seac/storage.py` | Per-agent rollout storage and old time-limit handling | Add a shared buffer with explicit bootstrap and trace masks |
| `robotic-warehouse/rware/warehouse.py` | Internal elapsed-step limit currently returns termination | Separate elapsed-time truncation from genuine termination |
| Routing environment | Requires environment-level `RewardType.INDIVIDUAL` | Aggregate team rewards in the learner; do not switch the environment to global rewards |
| Automatic service | Actions are ignored in `AUTO_PICKUP`, `AUTO_DELIVERY`, and `AUTO_DROP` | Exclude those decisions from actor losses, retaining rewards and value learning |
| Routing task assignment | Depends on completion order and available racks | Equal seeds pair initial randomness, not identical realized task sequences |
| Existing metrics | Whole-team stall proxy and completed-cycle summaries | Retain their names/meaning; add unfinished-task ages for evaluation |

Start with `rware-custom-5ag-routing-v2` and `assets/warehouse-10-6.map`. The routing setup has 199 local observation features and four actions: `NOOP`, `FORWARD`, `LEFT`, `RIGHT`. Read dimensions from spaces and assert the expected baseline rather than hard-coding dimensions throughout the model.

The README describes routing-mode infrastructure, not a demonstrated trained routing baseline. Existing SEAC results on different observations or rewards are contextual comparisons, not matched PPO/MAPPO controls.

## 3. Controlled experiment definitions

| Setting | `shared_ppo` | `mappo` |
|---|---|---|
| Actor | One shared network, local observation and optional local memory | Identical |
| Actor parameters | One parameter set and actor optimizer | Identical |
| Critic | One shared local critic evaluated separately for each robot | One centralized team critic evaluated once per environment |
| Training reward | Mean of original per-robot rewards | Identical |
| Policy loss | Per-robot clipped PPO | Identical |
| Execution | Local actor only | Local actor only |
| Simulator, seeds, interaction budget | Matched | Matched |

Use `r_team[t,e] = mean_i r_individual[t,e,i]`. Preserve individual rewards and every reward component in logs. A local critic predicts the same team return conditioned on that robot's local information; it is not trained on individual returns in this main comparison.

Optional later control: `shared_ppo_individual`, using each robot's own reward. Report this separately because it changes both learning signal and coordination incentives relative to the main pair. Do not use the existing `GlobalizeReward` wrapper, which sums rewards and would change scaling.

Implement feedforward versions first. Then add recurrence to **both** methods and compare feedforward PPO/MAPPO and recurrent PPO/MAPPO separately. Do not attribute a recurrent-versus-feedforward gain to the centralized critic.

## 4. Proposed code organization

New paths below are proposed, not existing commands or modules; `robotic-warehouse/tests/test_routing.py` already exists and will be extended.

| Path relative to repository | Responsibility |
|---|---|
| `seac/seac/train_shared.py` | Sacred entry point, configuration, collection, update, evaluation and checkpoint schedule |
| `seac/seac/shared_models.py` | Shared actor, local critic, centralized critic; separate actor/critic parameters |
| `seac/seac/shared_envs.py` | Vector runner with explicit reset/final-transition metadata and privileged-state extraction |
| `seac/seac/shared_storage.py` | Fleet rollout tensors, GAE, feedforward and contiguous-sequence batching |
| `seac/seac/shared_ppo.py` | Common clipped PPO updater; select local/team critic mode through configuration |
| `seac/seac/evaluate_shared.py` | Actor-only evaluation and per-run/per-robot metrics |
| `seac/seac/configs/shared_ppo_routing.yaml` | Local-critic defaults |
| `seac/seac/configs/mappo_routing.yaml` | Central-critic defaults with otherwise matched settings |
| `seac/tests/test_shared_*.py` | Learning/data-flow correctness tests |
| `robotic-warehouse/tests/test_routing.py` | Time-limit and simulator-invariance regression coverage |
| `scripts/run_shared_baselines.py` | Seeded experiment launcher after correctness gates pass |

Register only the new named configs in the new Sacred experiment. Resolve config paths relative to the script, not the caller's working directory. Avoid importing the legacy `train.py` experiment or A2C ingredient into the new entry point. Reuse compatible logging/statistics helpers without inheriting SEAC update semantics.

Only change simulator behavior where needed to correct termination semantics. Add a read-only state accessor if needed; do not expose it to the actor. Keep legacy `MultiAgentVecEnv` return signatures compatible, or isolate the new runner in `shared_envs.py`.

## 5. Environment transition contract

### 5.1 Time limits and final-state bootstrapping

Use one time-limit owner in new training runs: construct the simulator with `max_steps=None` and `max_inactivity_steps=None`, then use one outer `TimeLimit(500)`. Pass constructor overrides through the new environment factory. Also correct the simulator's internal `max_steps` branch to produce truncation for callers that retain an internal limit; keep genuine termination semantics separate and test the change.

Long-run evaluation must disable both the internal 500-step limit and the training wrapper. Stop externally after the evaluation budget, without resetting every 500 steps.

The new vector runner returns a structured batch with:

- `next_obs`: observations for the next policy call, including reset observations where applicable.
- `rewards_individual`, `terminated`, `truncated`, and `infos`.
- `final_obs` and `final_critic_input` captured **before** auto-reset, with a validity mask.
- `next_critic_input`: critic input corresponding to `next_obs`.
- Pre-action `decision_mask` derived from each robot's current task phase.

An environment reset applies to every robot in that environment. Never bootstrap a truncated transition from the next episode's reset observation. Preserve final episode metrics in `infos` across auto-reset.

### 5.2 Actions and memory

Store the policy's proposed action and its sampled log probability. A denied `FORWARD` remains the proposed action for PPO; replacing it with the resolver's `NOOP` corrupts the likelihood ratio.

At automatic-service states, execute `NOOP` and set actor/entropy/KL eligibility to zero. Do not train the actor to imitate that forced action. Keep the transition in the critic targets, GAE recursion, and recurrent observation sequence. Compute eligibility from the phase **before** the action, not the resulting phase.

Use the simulator's existing action semantics without adding occupancy masks or a new collision shield. If masking is added later, it must be a separate, matched experiment with stored masks.

Maintain separate hidden state for every `(environment, robot)` despite shared weights. Reset memory only on environment reset, not on pickup, delivery, task reassignment, or rollout-buffer boundaries.

## 6. Networks and information boundaries

### 6.1 Shared actor

Initial architecture: local observation → MLP(128, 128, ReLU) → optional GRU(128) → four categorical logits. Use one actor instance and batch the environment/robot dimensions for inference. Do not include robot identity embeddings, global state, centralized value output, or other robots' hidden states.

The recurrent actor consumes local observations at every simulator step, including service steps. Its hidden state is an implementation detail of that robot's history, not a communication channel.

### 6.2 Local critic

Use a separate MLP(128, 128) and scalar value head. Add its own GRU(128) in the recurrent configuration. It receives the same local observation stream as the actor and predicts team return for each robot. Actor and critic parameters/optimizers remain separate in both methods, avoiding a shared-trunk confound.

### 6.3 Centralized critic

For the initial fixed-map, five-robot comparison, prefer a simple MLP(256, 256) → scalar team value. Build a documented privileged vector containing:

- Static map/workstation and rack-home occupancy channels; actual dynamic shelf occupancy.
- Per-robot position, heading one-hot, loaded flag, phase one-hot, current pickup/workstation/return coordinates, task-presence flag, and observable-history fields needed for routing rewards such as blocked streak and previous distance/action.
- Any service or inactivity counters that actually affect dynamics or rewards; include fleet/map dimensions for normalization.

Normalize coordinates and bounded counters by documented constants. Derive the schema from the simulator fields; preserve the associations between each robot and its task. Do not include future assignments or RNG state. An artificial evaluation/training cutoff is not an input to a continuing-task value function.

Use stable robot ordering for this first centralized MLP. It is deliberately a fixed-fleet baseline; it does not establish permutation invariance or generalization to larger fleets. A set/graph critic is deferred until these baselines work. The centralized critic can remain feedforward in the recurrent-actor experiment because it receives privileged current state; disclose this architectural distinction.

Assert that state extraction does not advance simulation or consume RNG. Export the actor independently so evaluation runs without constructing or calling either critic. A batched actor server is acceptable for speed only if outputs equal independent local calls.

## 7. Rollout storage and return estimation

Use `T` rollout steps, `E` environments, `N` robots, `O` observation dimensions, and `H` hidden size.

| Tensor | Shape |
|---|---|
| Observations | `[T+1, E, N, O]` |
| Proposed actions, old log probabilities, decision masks | `[T, E, N]` |
| Individual rewards | `[T, E, N]` |
| Team rewards, terminated, truncated | `[T, E]` |
| Local values/advantages/returns | `[T(+1), E, N]`, with the extra slot only for values |
| Central values/advantages/returns | `[T(+1), E]`, with the extra slot only for values |
| Actor hidden states before observations | `[T+1, E, N, H]` in recurrent mode |
| Local-critic hidden states | Same indexing, separate storage |
| Central critic inputs | `[T+1, E, S]`, plus final inputs at resets |

Keep rollout-time values and log probabilities frozen throughout all update epochs. Precompute the correct bootstrap value for each transition, including final observations at truncations.

For either critic, define:

```text
bootstrap_mask[t] = 1 - terminated[t]
trace_mask[t]     = 1 - (terminated[t] OR truncated[t])
delta[t] = reward[t] + gamma * bootstrap_mask[t] * V(next_before_reset[t]) - V(current[t])
adv[t]   = delta[t] + gamma * gae_lambda * trace_mask[t] * adv[t+1]
return[t] = adv[t] + old_value[t]
```

At the final buffer boundary, initialize the unavailable future advantage to zero while retaining the next-state value bootstrap. At a true termination the bootstrap is zero. At a truncation bootstrap from the final state but stop the trace across the reset.

For local critics, broadcast the team reward and environment masks over robots and compute local advantages independently. For MAPPO, compute one team advantage per environment transition and broadcast it to actors. Do not replicate the centralized value loss N times.

For recurrent bootstrap evaluation, use the hidden state after consuming the current observation to evaluate the final next observation. Perform this without mutating the collector's live hidden state or consuming that observation twice.

## 8. PPO update and recurrent batching

For each eligible robot decision, use:

```text
ratio = exp(new_log_prob - old_log_prob)
policy_loss = -mean_valid(min(ratio * advantage,
                             clip(ratio, 1-epsilon, 1+epsilon) * advantage))
actor_loss = policy_loss - entropy_coef * mean_valid(entropy)
critic_loss = 0.5 * mean_valid((value - fixed_return_target)^2)
```

Use individual action ratios, not a product across the whole fleet. Normalize actor advantages once per rollout over eligible decisions with a numerical epsilon. Keep raw advantages and returns unchanged for diagnostics/value fitting. Critic eligibility excludes padding, not service steps. Skip actor updates safely if a batch has no eligible decisions.

Start with plain MSE value loss and no value normalization/clipping to keep the implementation auditable. If unstable return scale motivates normalization or clipping, implement it consistently in both baselines and record the change.

Shuffle individual samples only in feedforward mode. Recurrent mode must shuffle contiguous sequences while preserving `(environment, robot, time)` order within each sequence:

1. Store hidden state before each observation and episode reset masks.
2. Train on 32-step chunks; prepend up to 16 preceding observations for burn-in when available.
3. Start burn-in from its stored rollout hidden state, recompute under current weights without gradients, then train through the loss-bearing chunk.
4. Mask padding and apply internal episode resets. Each real transition belongs to one loss-bearing chunk per epoch; burn-in is not extra training data.
5. Keep old log probabilities from collection. Never regenerate the denominator with updated weights.

Stored hidden states under the previous policy are an approximation after updates; burn-in reduces, but does not eliminate, that mismatch. Monitor KL and keep updates small. Retain pre-rollout history if burn-in crosses a rollout boundary; otherwise record the shorter burn-in rather than inventing history.

Log approximate KL, clip fraction, entropy, actor/critic gradient norms, explained variance, valid-decision fraction, and actual epochs completed. Stop remaining actor epochs when rollout-level estimated KL exceeds the configured threshold. Define critic updates independently so the policy stopping rule is unambiguous.

## 9. Starting configuration and proposed CLI

These values are starting points for validation, not established optimal hyperparameters.

```yaml
env_name: rware-custom-5ag-routing-v2
seed: 0
num_env_steps: 2000000
time_limit: 500
num_envs: 8
rollout_steps: 256
device: auto
method: shared_ppo                 # mappo changes critic mode
reward_mode: team_mean
recurrent: false                   # true after feedforward gates
hidden_size: 128
gamma: 0.99
gae_lambda: 0.95
actor_lr: 0.0003
critic_lr: 0.0003
ppo_epochs: 4
num_minibatches: 4
clip_epsilon: 0.2
entropy_coef: 0.01
max_grad_norm: 0.5
target_kl: 0.02
normalize_advantage: true
value_loss: mse
sequence_length: 32
burn_in: 16
eval_interval_env_steps: 250000
save_interval_env_steps: 250000
```

Use constant learning rates initially. Count one joint warehouse transition as one environment step: a rollout uses `T * E` environment steps and `T * E * N` agent transitions. Log both, plus wall time and hardware. All schedules and budgets use actual environment steps, never update indices. Log overshoot of a requested budget if finishing a complete rollout.

The current runner is synchronous; `num_envs=8` does not imply eight CPU workers. Benchmark collection versus optimization time before changing execution infrastructure. If memory requires a smaller batch, use the same batch setting for both methods.

Implement this CLI contract, then document it in the README. These commands will only work after the new files exist:

```bash
cd seac/seac
../../.venv/bin/python train_shared.py with shared_ppo_routing seed=0 num_env_steps=8192
../../.venv/bin/python train_shared.py with mappo_routing seed=0 num_env_steps=8192
../../.venv/bin/python train_shared.py with shared_ppo_routing seed=0 num_env_steps=2000000
../../.venv/bin/python train_shared.py with mappo_routing seed=0 num_env_steps=2000000
../../.venv/bin/python train_shared.py with shared_ppo_routing recurrent=True seed=0 num_env_steps=2000000
../../.venv/bin/python train_shared.py with mappo_routing recurrent=True seed=0 num_env_steps=2000000
```

Use the repository's existing working virtual environment; do not make a dependency upgrade part of this algorithm change. Record the actual dependency versions and resolved configuration in each run.

### 9.1 Two-GPU execution contract

Verified on 2026-09-13: two NVIDIA RTX A2000 GPUs, 6,138 MiB each;
`.venv` has PyTorch 2.5.1+cu121, Gymnasium 1.3.0 and Sacred 0.8.7,
and PyTorch reports two available CUDA devices. These are observed versions,
not instructions to upgrade dependencies. An existing SEAC-GRU job was running;
do not stop it or count its results as PPO/MAPPO validation.

Implement `scripts/run_shared_baselines.py` as a dedicated launcher for only
`shared_ppo_routing` and `mappo_routing`. Follow the existing
`scripts/run_comparison.py` subprocess pattern without changing its A2C/SEAC
method list. No DDP, DataParallel, shared optimizer, or cross-model gradients:
these GPUs run independent experiments concurrently.

- Default physical assignment: Shared PPO on GPU 0, MAPPO on GPU 1. Require
  two distinct GPU selectors via `--gpus 0 1`; reject duplicates and invalid
  selectors before starting either child. For predictable physical selection,
  resolve indices to GPU UUIDs using `nvidia-smi` and pass the selected UUID as
  each child's `CUDA_VISIBLE_DEVICES`. Document this physical-index convention
  and reject a conflicting inherited visibility restriction instead of silently
  selecting a GPU outside that restriction.
- Each child sees exactly one GPU and receives `device=cuda:0`, including the
  child assigned physical GPU 1. Record both the physical index/UUID and logical
  device. Never pass an empty `CUDA_VISIBLE_DEVICES` value.
- Run both methods for a seed concurrently, wait for both, then start the next
  seed. Recurrent experiments are a separate invocation/output root after their
  gates pass; at most one child per selected GPU within this launcher.
- Use the same seed, interaction budget, rollout size, environment count,
  recurrence mode and evaluation schedule for the pair. Start with one PyTorch
  CPU thread per child (`torch.set_num_threads(1)`) and set `OMP_NUM_THREADS=1`
  and `MKL_NUM_THREADS=1`. The environment runner remains synchronous.
- Resolve repository paths from `__file__`, use the repository `.venv/bin/python`
  by default with an explicit `--python` override, and launch from `seac/seac`.
  Register named configs relative to the trainer file. Put Sacred observers,
  metrics, TensorBoard logs, evaluation rows and checkpoints under that child's
  unique run directory, not a shared relative `results/sacred` directory.
- Output layout: `<output>/<method>/seed_<seed>/<attempt>/`. Preflight all target
  directories, refuse existing attempts, then create directories exclusively.
  Save `launch.json` with the exact executed argv/environment, PID, timestamps,
  device identity and eventual exit status; stream each child to `train.log`.
- `--dry-run` prints the exact commands, GPU mapping and destinations without
  requiring available CUDA or creating output directories. Actual execution
  checks CUDA availability in each child's visibility context and fails instead
  of silently falling back to CPU.
- On child failure, stop its sibling and return nonzero. On SIGINT/SIGTERM or a
  spawn error, terminate and reap only children launched by this invocation,
  with bounded wait and kill fallback; close logs and record interrupted status.
  Never terminate unrelated training processes. Surface occupied-GPU processes
  before launch; an explicit `--allow-busy` permits intentional sharing.

Proposed commands, **not runnable until implementation is complete**, from the
repository root:

```bash
.venv/bin/python scripts/run_shared_baselines.py --gpus 0 1 --seeds 0 --num-env-steps 8192 --attempt smoke --dry-run
.venv/bin/python scripts/run_shared_baselines.py --gpus 0 1 --seeds 0 --num-env-steps 8192 --attempt smoke
.venv/bin/python scripts/run_shared_baselines.py --gpus 0 1 --seeds 0 1 2 --num-env-steps 2000000 --attempt ff_screen --output results/shared_baselines
# Only after recurrent correctness gates:
.venv/bin/python scripts/run_shared_baselines.py --gpus 0 1 --seeds 0 1 2 --num-env-steps 2000000 --recurrent --attempt gru_screen --output results/shared_baselines
```

Measure collection/update wall time and peak CUDA memory for both children.
The small MLP and Python simulator may limit GPU utilization; two GPUs improve
concurrent experiment throughput, not necessarily a single model's speed.
If 6 GB is insufficient, lower `num_envs`/`rollout_steps` equally through explicit
launcher overrides, record the resulting batch sizes and rerun both methods.
Do not silently shrink a single method or enable mixed precision in this milestone.

Launcher acceptance: dry-run shows one method per GPU; a subprocess test uses
short stand-in children to verify concurrent lifetime, isolated visibility/output,
exit propagation, overwrite refusal and signal cleanup without needing GPUs.
Then run both real 8,192-step GPU smoke jobs together, checking distinct physical
GPU placement, finite losses, changed weights, independent checkpoints and zero
exit codes. This is a readiness check, not evidence of learned routing quality.

## 10. Implementation stages and acceptance gates

### Stage 0 — freeze scope and protect existing work

- [ ] Start implementation from the recorded baseline plus this plan; record the
  final implementation commit in run metadata.
- [ ] Preserve `train.py`, `a2c.py`, `shared_a2c.py`, `teams.py`, the four existing
  phase-2 configs, `run_comparison.py`, existing checkpoints and result directories.
- [ ] Keep changes to new PPO/MAPPO modules/configs/tests/launcher and README,
  plus the narrow warehouse time-limit correction and its regression coverage.
- [ ] Do not implement the older four-algorithm plan again. Do not add a third
  algorithm, CCPD scaffolding, a dependency migration or multi-GPU model training.

### Stage A — simulator/collector contract

- [ ] Correct elapsed-time truncation; make the new factory's single time-limit ownership explicit.
- [ ] Preserve final local observations and final centralized inputs through auto-reset.
- [ ] Add pre-action service masks and read-only centralized inputs.
- [ ] Verify action/observation dimensions and reward component sums.
- [ ] Run existing relevant simulator tests after the termination change.

Gate: controlled termination/truncation fixtures pass; actions, rewards and physical transitions match the old simulator before the cutoff. No unexplained simulator regression.

### Stage B — feedforward shared PPO

- [ ] Implement one actor, one local critic, shared buffer, GAE and clipped updates.
- [ ] Implement deterministic/stochastic actor-only evaluation and checkpoints.
- [ ] Run the 8,192-step smoke configuration; verify finite outputs and actual actor/critic parameter changes.
- [ ] After implementation readiness, run one 2M-step diagnostic training seed and inspect task-cycle metrics and trajectories.

Code gate: correctness tests and smoke training pass. Research gate, before interpreting Stage E: repeatable evidence of learning beyond frozen initialization on diagnostic evaluation. A short-run failure is a debugging signal, not proof the algorithm cannot learn.

### Stage C — feedforward MAPPO

- [ ] Reuse the same actor/updater/collector; replace local value estimation with centralized team value estimation.
- [ ] Verify critic inputs are training-only and centralized value loss is not multiplied by fleet size.
- [ ] Repeat smoke runs with matched settings; perform matched diagnostic runs after implementation readiness.

Gate: MAPPO trains and deploys using the actor alone. It need not outperform PPO at this stage.

### Stage D — recurrent versions

- [ ] Add independent actor memory per robot and separate local-critic memory where required.
- [ ] Implement contiguous sequence batches, burn-in, reset/padding masks and recurrent bootstrap handling.
- [ ] Repeat smoke tests and diagnostic runs for both methods.

Gate: sequence replay reproduces collection log probabilities before updates; hidden states never cross robots or resets; both recurrent runs complete without numerical failures.

### Stage D2 — implementation readiness and parallel GPU validation

- [ ] Add the launcher from §9.1 and validate child concurrency and cleanup with
  stand-in subprocess tests, including one failed launch after its sibling starts.
- [ ] Run the targeted existing regressions and new PPO/MAPPO correctness tests:

  ```bash
  .venv/bin/python -m pytest seac/tests/test_comparison.py robotic-warehouse/tests/test_routing.py robotic-warehouse/tests/test_task_manager.py
  .venv/bin/python -m pytest seac/tests/test_shared_*.py
  ```

- [ ] Complete CPU smoke training for both methods with `device=cpu`.
- [ ] Complete paired GPU smoke training with independent output directories,
  checking both physical GPUs, saved/reloaded actor outputs, finite diagnostics
  and real actor/critic weight updates. Repeat for recurrent variants after Stage D.
- [ ] Record smoke results and measured memory/timing; document exact CLI examples
  in README only when they work. Commit and push the scoped implementation.

Gate: runnable training and evaluation for both methods, reproducible launch
metadata, passing correctness checks and successful concurrent two-GPU smoke runs.
This completes implementation readiness. The long research runs in Stage E are
subsequent experiments, not a condition for committing working training code.
Do not automatically launch millions of steps as part of a code smoke check.

### Stage E — controlled baseline study

- [ ] Screen the four configurations for 2M steps with seeds 0, 1, 2; inspect curves, not only final reward.
- [ ] Allocate the same tuning budget to both methods. If necessary, try actor LR {1e-4, 3e-4} and entropy {0.003, 0.01} using validation seeds only.
- [ ] Train the selected PPO and MAPPO configurations for 10M steps using five independent training seeds, 0–4.
- [ ] Extend both to 40M only if learning curves and resources justify a longer comparison; commit to the rule before viewing final test results.
- [ ] Evaluate final-budget checkpoints and separately report validation-selected checkpoints.

Gate: complete comparable results, resource costs and uncertainty estimates; explain failures. Select the backbone from evidence. If PPO matches or exceeds MAPPO, retain that finding and diagnose centralized value learning before assuming MAPPO is the stronger foundation.

## 11. Essential correctness tests

These tests target algorithmic failures, not incidental implementation details.

| Test | Required evidence |
|---|---|
| Analytic GAE examples | Exact expected returns for ordinary steps, true termination, truncation and buffer boundaries |
| Auto-reset distinction | Deliberately different final/reset observations demonstrate bootstrap uses the final observation |
| Shared parameters | One actor parameter set; experience from different robots contributes to the same actor update |
| Decentralized execution | Actor runs without critic/global state; batched and separate robot inference agree |
| PPO likelihoods | Before updating, recomputed log probabilities match collection and ratios are approximately one |
| Denied movement | Original proposed action/log probability survives collision resolution |
| Automatic service | Zero actor/entropy contribution, nonzero applicable critic targets, memory still advances |
| Recurrent resets | Robot A's reset/history cannot alter robot B's hidden state; reset boundaries inside chunks are respected |
| Recurrent alignment | Chunked replay with stored initial state matches full-sequence replay under frozen weights |
| Central critic reduction | One value target per team transition, with correct reward/advantage broadcasting |
| Privileged accessor | Reading critic input leaves subsequent seeded simulator transitions unchanged |
| Checkpoint export | Reloaded actor produces equal logits/actions and starts with correctly initialized memory |

Run a CPU smoke path even if full training uses CUDA. Avoid expensive broad reruns after these checks pass unless a concrete regression warrants them.

## 12. Evaluation protocol

Use the original five-robot map first to establish learning. Keep training, validation and test seed sets separate. Suggested validation seeds: 1000–1019; test seeds: 2000–2049. Record seed lists before final comparisons.

Evaluate two regimes:

1. **Matched 500-step episodes:** 50 test seeds per trained checkpoint for compatibility with the development setting.
2. **Continuous 10,000-step runs:** at least 20 test seeds per checkpoint, with the internal/outer 500-step limits disabled. Report the full-run result and a separately defined post-1,000-step warm-up result. Do not reset memory or tasks after warm-up.

Use deterministic argmax evaluation as the primary mode for both methods, and report a separate stochastic-policy sensitivity check. Do not choose whichever mode makes each method look best after seeing test results.

Report:

- Completed full pickup–delivery–return cycles per 1,000 joint environment steps, as the primary throughput metric.
- Deliveries, pickups, completed cycle-time mean and p95, and per-robot completed cycles.
- Unfinished task count, current cycle age distribution/max, and fraction of robots completing no cycle over the measurement window.
- Robot-blocked/denied-forward frequency per navigation decision; separate physical shelf/boundary denials where available.
- Existing whole-team `deadlock_events`, explicitly labeled as a limited stall proxy.
- Reward component totals, plus actor inference latency/fleet step and training wall time.

Collect raw cycle completion durations for p95; episode means cannot recover a percentile. Track unfinished ages even when no cycle completes. Preserve existing wait-step semantics instead of silently counting turns as waits.

Compare with a frozen initialized actor, a random-action policy and a documented load-aware greedy routing reference if implemented. A legacy SEAC comparison needs rerunning under matched observations, rewards, evaluation and limits; otherwise label it as historical context.

For uncertainty, treat independently trained models as the primary replication units. Report five-seed means and intervals, and paired test-scenario differences; use hierarchical resampling if aggregating training and evaluation seeds. Hundreds of episodes from one trained model are not hundreds of training replications.

Equal task-manager seeds do not imply identical task streams across policies because completion order affects assignment. State this limitation. Keep the current task manager for this milestone and report results over many seeds; a policy-independent demand benchmark is a later, separately controlled change.

This first fixed-map comparison establishes a baseline, not generalization or SOTA. Held-out layouts, fleet sizes, stronger external solvers and matched motion/information constraints are required for subsequent publication claims.

## 13. Logging, checkpoints and reproducibility

Each run saves resolved config, source commit plus dirty diff when applicable, map hash, seed, software/hardware versions, environment/agent step counts, evaluation seed lists, model parameter counts, losses, task metrics and timing.

Save one actor state dict, the selected critic state dict, optimizer states, update/global-step counters, RNG states and any preprocessing statistics. Save at rollout boundaries and protect run directories from accidental overwrite. Maintain `last` and validation-selected `best` checkpoints with a documented selection metric.

Without full simulator snapshots, resuming restarts environments: this preserves model/optimizer progress but is **not an exact trajectory continuation**. Document that limitation; do not implement simulator cloning solely to make baseline resumes exact. Inference export requires only actor weights, architecture/observation schema and preprocessing metadata.

Evaluation must use a separate environment/RNG stream so it does not perturb subsequent training task assignment. Save raw per-evaluation rows, not just dashboard averages.

## 14. Boundary after PPO/MAPPO

CCPD is outside this implementation. Do not add its flag, modules, hooks, replay or losses; unknown configuration keys must fail validation. Provide stable actor `act`/`evaluate_actions` interfaces and sequence batches containing local observations, proposed actions, old log probabilities, task phases, episode identifiers, environment/robot indices and masks. Keep raw rewards and advantages available.

Do not create successful-event replay, event weights, peer losses or counterfactual rollouts now. The earlier research plans remain separate. This milestone should make the future auxiliary mechanism easy to add without changing baseline semantics.

Proceed only after:

- [ ] Both methods pass the correctness gates and have completed the matched baseline study.
- [ ] Evaluation exposes whether congestion/coordination is a meaningful remaining limitation.
- [ ] The chosen backbone has stable training and an actor-only deployment path.
- [ ] A frozen baseline config/checkpoint and complete metric report are archived.
- [ ] Future CCPD experiments will include the unchanged backbone and a matched extra-training control.

If neither baseline learns reliable task cycles, fix the training/data/environment interface before adding auxiliary losses. If centralized training offers no benefit, investigate critic representation and value error rather than treating the method name as evidence of strength.

## 15. Source references

Implementation facts were checked against the local checkout at the pinned commit, including a seeded environment reset and installed CUDA availability. No PPO/MAPPO training has been run:

- [README and simulator configuration](https://github.com/s-w3i/seac-rware/blob/56238fc8b699e257effaa48a8638f153faec69c0/README.md)
- [Existing trainer](https://github.com/s-w3i/seac-rware/blob/56238fc8b699e257effaa48a8638f153faec69c0/seac/seac/train.py)
- [Vector runner](https://github.com/s-w3i/seac-rware/blob/56238fc8b699e257effaa48a8638f153faec69c0/seac/seac/envs.py)
- [Rollout storage](https://github.com/s-w3i/seac-rware/blob/56238fc8b699e257effaa48a8638f153faec69c0/seac/seac/storage.py)
- [Statistics and wrappers](https://github.com/s-w3i/seac-rware/blob/56238fc8b699e257effaa48a8638f153faec69c0/seac/seac/wrappers.py)
- [Warehouse dynamics and termination](https://github.com/s-w3i/seac-rware/blob/56238fc8b699e257effaa48a8638f153faec69c0/robotic-warehouse/rware/warehouse.py)
- [Environment registration](https://github.com/s-w3i/seac-rware/blob/56238fc8b699e257effaa48a8638f153faec69c0/robotic-warehouse/rware/__init__.py)
- [Current routing training config](https://github.com/s-w3i/seac-rware/blob/56238fc8b699e257effaa48a8638f153faec69c0/seac/seac/configs/rware_custom_5ag_routing.yaml)
