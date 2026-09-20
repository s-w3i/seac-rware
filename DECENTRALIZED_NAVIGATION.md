# Decentralized warehouse navigation

This isolated checkout implements a new experimental protocol. It does not modify the original workspace or the completed CCPD investigation. Training performance and generalization are not established by code validation.

## Run

Validation only (the default; four 4,096-step CPU smoke checks, never full training):

```bash
/home/utar/seac-rware/.venv/bin/python /home/utar/seac-rware-decentralized-navigation/scripts/run_decentralized_navigation.py
```

User-launched full experiment:

```bash
/home/utar/seac-rware/.venv/bin/python /home/utar/seac-rware-decentralized-navigation/scripts/run_decentralized_navigation.py --train
```

Add `--dry-run` to inspect either mode without creating outputs or probing GPUs. The full campaign is twelve fresh runs: local / local+CCPD / communicating / communicating+CCPD, each with seeds 0–2 and 20M environment steps. Rollout rounding can exceed an individual budget by less than 2,048 steps. A fleet environment step contains all five robot decisions.

The launcher configures the isolated simulator import explicitly, validates the code first, then runs four trainers concurrently per seed: Local MAPPO on physical GPU 0, and Local MAPPO + CCPD, Communicating MAPPO, and Communicating MAPPO + CCPD together on physical GPU 1. Seeds 0, 1, and 2 run in successive waves; each wave waits for all its runs to finish. Both assigned GPUs must be free of unrelated compute jobs before a wave starts. It waits when an assigned GPU is busy and respects `CUDA_VISIBLE_DEVICES`. Existing unrelated jobs are not terminated. A failed owned job stops its current owned wave and records the failure. Completed stages are reused only after configuration, source and artifact hashes match. Partial or incompatible artifacts require explicit recovery; they are never silently counted as complete or overwritten. Use a different `--output` for a deliberately different experiment.

Output defaults to `/home/utar/seac-rware/results/decentralized_navigation/`: `status.json`, `pipeline.log`, validation receipts, per-run training and validation records, per-episode evaluation JSONL, actor exports, `comparison.md`, `next_steps.md`, and `summary.json`. `validated` means code checks passed, not that the research campaign ran. `complete` means training, evaluation and reports finished.

## Policy and information boundary

The actor receives a fixed 7×5×5 egocentric grid plus 18 own-robot features. The grid contains in-bounds, shelf occupancy, robot presence, and four robot-relative headings. Scalars contain carrying, six task phases, relative goal bearing/distance, previous action, movement success/denial, stationary age (clipped at 64), and cycle age (clipped at 1,000). Goal distance is `d/(1+d)` in grid cells. No actor input contains absolute position, absolute heading, robot/layout identity, global request flags, shortest-path values, or evaluation seeds.

Two convolution layers and a scalar encoder feed a 128-unit GRU. All robots share weights but retain separate memory and sampling RNGs. The centralized critic uses masked map convolutions and pooled robot features; it is not exported or required for inference. Existing reward shaping, including simulator distance calculations, remains training feedback. Grid task service remains automatic under the existing rules; no new navigation action mask, route planner, yielding rule or escape controller is introduced.

The packet payload is a sender's own status. Frame metadata is converted to relative position, heading and goal features before reaching the actor. One broadcast is sent per fleet step per robot to recipients within Chebyshev distance two at transmission. Delivery takes one step, packets expire after age two, and training drops 10% of recipient copies using an independent RNG. The newest still-valid packet is retained after a loss. Messages never cross environment resets. A small MLP and one attention head aggregate the inbox without robot identifiers. An empty inbox gives exactly zero context.

Packet content is structured, not a separately learned symbolic negotiation language. Coordination actions are learned through RL. PPO stores the actual delivered packets and masks with observations and never resamples channel noise during probability reconstruction.

## Actor-only deployment

Each training directory exports `actor.pt`, `observation_spec.json`, and a copy of `navigation_policy.py`. The latter imports only NumPy, PyTorch and the standard library, not Gym or the simulator.

```python
import numpy as np
from navigation_policy import PolicyRuntime

policy = PolicyRuntime('actor.pt', device='cpu')
memory = None
rng = np.random.default_rng(17)  # separate generator for each AGV
action, memory = policy.act(local_observation, inbox, memory, rng)
```

`local_observation` is the 193-float sensor-adapter output; `inbox` is zero or more 20-float preprocessed packets. `packet_features` converts sender status and receiver pose into packet features, rejecting future/expired packets. Pose metadata is only for coordinate conversion. The runtime returns one of wait/forward/left/right and the next 128-float memory. Reset memory on episode/mission reset, not at every new rack task. Probabilities are tested against batched simulator inference. A real sensor/actuator adapter, network transport and continuous robot dynamics are outside this grid implementation.

## Multi-layout readiness

`assets/navigation_layouts.json` contains the original registered warehouse expressed with `.`, `x`, and `g`. Each entry has an ID, family, path, split, positive sampling weight and geometry hash. The loader checks dimensions, rack/workstation counts, hashes and loaded routes. `load_layouts(..., mode='cross_layout', training_families=...)` rejects overlapping families; reflections/rotations are checked to share the same family. Entries marked `engineering_fixture: true` are rejected for cross-layout evaluation claims.

`NavigationEnvs(layouts, seed, horizons, ...)` samples from the supplied training layouts at reset, pads maps to the manifest's maximum dimensions, and clears robot memory through reset flags plus fresh communication buffers. The actor and critic weights do not depend on layout dimensions or robot ordering. Engineering checks exercise different map sizes and three/seven robots. The declared launcher intentionally permits only the original layout for full training. A future cross-layout campaign must explicitly declare its manifests and evaluation protocol rather than silently extending this experiment. Shape tests are not generalization results.

Training uses eight environments with horizons 500, 500, 2,000, 2,000, 10,000, 10,000, 10,000, 10,000 and 256-step rollouts. Backpropagation remains limited to 32-step sequences with 16-step burn-in. After each optimizer update, carried actor memory is refreshed through that recorded burn-in under the new weights; this preserves probability reconstruction on the next rollout without resetting long-episode memory. Layout, spawn, task, action, channel and auxiliary-selection RNG streams are independent. Generated training seeds occupy a separate namespace. Evaluation rejects reserved seeds 3000–3049.

## Evaluation and interpretation

Validation every 2M steps uses stochastic execution: seeds 1000–1019 for 500 steps and 1000–1007 for 10,000. Best selection is lexicographic: fewer long-run fleet completion gaps ≥500, higher long-run throughput, then earlier checkpoint. Final checkpoints are primary; validation-selected best results are reported separately.

Final evaluation uses seeds 2000–2049 at 500 and 10,000 steps, plus 20 changed-task and 20 changed-start cases at 10,000 steps. All stochastic cases use three independent action-RNG realizations. Deterministic short/long results are separate. Communication long-run delivery conditions include loss 0%, 10%, 30% and 100%; 10% is the primary matched condition. Best checkpoints receive stochastic reference short/long suites only.

Counterfactuals keep starts/headings or initial tasks constant by separating spawn/task RNG streams. Completion-dependent future task assignments can diverge; only the stated initial conditions are guaranteed identical. Counterfactual throughput is compared with the same 20 reference episode IDs. The environment RNG, channel RNG and action RNG identities are retained in results.

Reports reuse the existing rate aggregation and paired hierarchical bootstrap: 10,000 draws over matched training seeds then matched episode IDs. Action-RNG replicates are retained together within each episode block. Three training seeds give limited precision and intervals are pointwise. A fleet failure means any completion gap ≥500, measured between completion timestamps and including the episode start/end boundaries, not just an old task at the endpoint. A gap reaching 500 is counted once even if it lasts longer; a completion at exactly 500 still meets the threshold. The old deadlock counter remains explicitly a team-stall proxy. Completed-cycle means are null when no cycle completes.

The communication screen requires each seed to meet the 1% fleet-failure target, the 2% throughput tolerance, non-worsening unfinished-age indicators and the 5% counterfactual tolerance. It additionally requires either ≥3% mean long-run throughput gain with an interval excluding zero, or ≥2 percentage points lower failure frequency with an interval supporting improvement. CCPD is assessed within each communication setting and must reduce short- and long-run conflicts in every seed without failing those safeguards. Packet-loss/outage behavior is reported separately and must be inspected before deployment choices. A failed or inconclusive screen is not equivalence. These comparisons measure net CCPD usefulness, not success-selection causality.

No script automatically launches another budget, expands layouts or uses the reserved final seeds based on its recommendation.
