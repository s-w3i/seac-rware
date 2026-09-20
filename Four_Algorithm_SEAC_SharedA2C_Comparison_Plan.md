# Four-Algorithm Comparison Implementation Plan

## Goal

Build and compare a clean 2×2 baseline matrix on the frozen Phase-2 warehouse environment.

| Policy structure | No memory | GRU memory |
|---|---|---|
| Independent policies + SEAC | **SEAC** | **SEAC-GRU** |
| One shared policy + pooled A2C | **Shared-A2C** | **Shared-A2C-GRU** |

The experiment isolates two factors:

1. **Policy sharing:** independent SEAC policies vs one shared policy.
2. **Memory:** feed-forward policy vs recurrent GRU policy.

Do not change the environment, observation, reward, action space, hidden width, optimizer, rollout length, or training budget while running this comparison.

---

## 1. Freeze the Common Phase-2 Setup

Use the exact current Phase-2 routing environment for all four methods.

Common settings:

- environment: `rware-custom-5ag-routing-v2`
- robots: 5
- observation: existing Phase-2 flattened observation
- action space: NOOP / FORWARD / LEFT / RIGHT
- episode limit: 500
- training algorithm family: A2C
- learning rate: 3e-4
- gamma: 0.99
- entropy coefficient: 0.01
- value loss coefficient: 0.5
- gradient clipping: 0.5
- num_processes: 4
- num_steps: 5
- training budget: 40M environment steps

Do not add CNN, attention, new reward terms, new observation fields, curriculum, or new heuristics yet.

---

## 2. Algorithm A — SEAC

This is the already-completed Phase-2 baseline.

```text
Robot 1 → Policy 1
Robot 2 → Policy 2
Robot 3 → Policy 3
Robot 4 → Policy 4
Robot 5 → Policy 5
              ↕
         SEAC losses
```

Each robot has its own `Policy`, optimizer, and rollout storage.

Use:

```yaml
algorithm:
  recurrent_policy: false
```

Preserve checkpoint, config, logs, Git commit, seed, and evaluation outputs.

---

## 3. Algorithm B — SEAC-GRU

Purpose: test whether memory improves SEAC while keeping independent policies.

```text
Robot 1 → GRU Policy 1
Robot 2 → GRU Policy 2
Robot 3 → GRU Policy 3
Robot 4 → GRU Policy 4
Robot 5 → GRU Policy 5
                 ↕
            SEAC losses
```

The current `Policy` already supports a GRU.

### Important recurrent-SEAC requirement

Do not naively use another policy's GRU hidden state when evaluating its trajectory.

For recurrent SEAC, policy i evaluating robot/policy j's trajectory needs a recurrent state produced by policy i.

Implement a cross-policy hidden-state matrix:

\[
H_{ij}
\]

where i is the evaluating policy and j is the robot observation stream.

`H_ij` means: recurrent state of policy i after observing robot j's history.

For own acting, `H_ii` is the normal hidden state.

For SEAC cross-policy evaluation, use `H_ij` as recurrent context.

Add a helper that can advance recurrent state on an observation stream without requiring the policy to control that robot.

Before full training, verify recurrent replay numerically on a deterministic synthetic sequence.

---

## 4. Algorithm C — Shared-A2C

Purpose: test whether homogeneous robots can simply learn one shared policy instead of independent SEAC policies.

```text
Robot 1 ─┐
Robot 2 ─┤
Robot 3 ─┼── ONE Policy
Robot 4 ─┤
Robot 5 ─┘
```

No SEAC loss. No importance sampling. No GRU.

Create a new trainer, preferably:

```text
seac/seac/shared_a2c.py
```

Keep original `a2c.py` intact for SEAC.

`SharedA2C` owns:

```python
self.model
self.optimizer
self.storages
```

There is exactly:

```text
1 Policy object
1 optimizer
N rollout storages
```

Average A2C loss across robots:

\[
L_{policy} = rac1N\sum_i L_i^{policy}
\]

\[
L_{value} = rac1N\sum_i L_i^{value}
\]

\[
L_{shared}
=
L_{policy}
+
c_vL_{value}
-
c_eH
\]

Perform one optimizer step per update.

---

## 5. Algorithm D — Shared-A2C-GRU

Purpose: test whether memory improves the shared-policy baseline.

```text
Robot 1 ─┐
Robot 2 ─┤
Robot 3 ─┼── ONE shared GRU policy
Robot 4 ─┤
Robot 5 ─┘
```

Weights are shared:

\[
	heta_1=\cdots=	heta_N=	heta
\]

Hidden states are separate:

\[
h_1
eq h_2
eq\cdots
eq h_N
\]

Use the same `SharedA2C` trainer, changing only:

```yaml
algorithm:
  recurrent_policy: true
```

Verify that one robot's hidden state can change without modifying another robot's hidden state and that episode termination resets the corresponding hidden state.

---

## 6. Four Explicit Configs

Suggested names:

```text
rware_phase2_seac.yaml
rware_phase2_seac_gru.yaml
rware_phase2_shared_a2c.yaml
rware_phase2_shared_a2c_gru.yaml
```

| Config | Trainer | recurrent_policy |
|---|---|---:|
| SEAC | `A2C` | false |
| SEAC-GRU | `A2C` | true |
| Shared-A2C | `SharedA2C` | false |
| Shared-A2C-GRU | `SharedA2C` | true |

All other hyperparameters stay fixed.

---

## 7. Training Loop

Avoid maintaining duplicate training loops.

Prefer a small team/trainer abstraction:

```python
team.act(step)
team.insert(...)
team.compute_returns()
team.update()
team.after_update()
team.save(...)
```

Possible implementations:

```text
SEACTeam
SharedA2CTeam
```

Both should use the same environment interaction loop, logging, checkpointing, and evaluation code.

---

## 8. Evaluation

SEAC / SEAC-GRU:

```text
robot i → policy i
```

Shared-A2C / Shared-A2C-GRU:

```text
every robot → same policy
```

For GRU methods, keep one hidden-state tensor per robot. Never share one GRU hidden state across all robots.

---

## 9. Metrics

Primary metrics:

- cycles per 1000 steps
- mean cycle time
- deliveries per 1000 steps
- movement-denied rate
- conflict-attempt rate
- deadlock-event rate
- wait-step ratio
- path stretch

Learning metrics:

- actor loss
- value loss
- entropy
- gradient norm
- training FPS
- update time

SEAC-only metrics:

- SEAC policy loss
- SEAC value loss
- importance-sampling ratio

Compute metrics:

- parameters per policy
- total deployed parameters
- number of policy networks
- inference latency
- GPU memory

---

## 10. Seeds and Evaluation Protocol

Development:

- one seed for smoke testing

Research comparison:

- same 5 training seeds for all algorithms
- e.g. 0, 1, 2, 3, 4
- same environment/task seeds
- 40M environment steps per seed

Periodic evaluation can remain small for monitoring.

Final comparison:

- at least 100 evaluation episodes/instances per training seed
- identical fixed evaluation seeds across all four methods
- report mean and 95% confidence interval across independent training seeds

---

## 11. 2×2 Scientific Analysis

Effect of GRU under SEAC:

\[
\Delta_{GRU|SEAC}
=
Score(SEAC	ext{-}GRU)-Score(SEAC)
\]

Effect of GRU under Shared-A2C:

\[
\Delta_{GRU|Shared}
=
Score(Shared	ext{-}GRU)-Score(Shared)
\]

Effect of shared policy without GRU:

\[
\Delta_{Shared|MLP}
=
Score(Shared)-Score(SEAC)
\]

Effect of shared policy with GRU:

\[
\Delta_{Shared|GRU}
=
Score(Shared	ext{-}GRU)-Score(SEAC	ext{-}GRU)
\]

This isolates whether memory matters, parameter sharing matters, and whether SEAC still adds value after memory is controlled.

---

## 12. Required Tests Before Full GPU Runs

- all Shared-A2C robots reference the exact same model object
- only one optimizer exists for Shared-A2C
- gradients from different robots update the same shared parameters
- Shared-A2C contains no SEAC loss
- separate GRU hidden state per robot
- done masks reset recurrent state correctly
- non-recurrent SEAC still behaves as before
- SEAC-GRU uses target-policy recurrent context for cross-agent evaluation
- all four methods save and restore correctly
- Shared-A2C with N=1 approximately matches normal A2C

---

## 13. Exact Implementation Order

```text
DONE: Phase-2 SEAC baseline
        ↓
1. Freeze/tag current baseline
        ↓
2. Implement Shared-A2C
        ↓
3. Smoke-test Shared-A2C
        ↓
4. Enable Shared-A2C-GRU
        ↓
5. Test separate hidden states + reset behavior
        ↓
6. Implement correct recurrent cross-policy state handling
        ↓
7. Enable SEAC-GRU
        ↓
8. Smoke-test all four methods
        ↓
9. Run one common seed for all four
        ↓
10. Run five-seed full 40M experiments
        ↓
11. Evaluate using one fixed evaluation suite
        ↓
12. Produce 2×2 results table + learning curves
```

---

## 14. Stop Before the Next Research Phase

Do not add CNNs, attention, curriculum, or Population-SEAC until this table exists:

| Method | Cycles/1k | Cycle Time | Conflict Rate | Deadlock Rate | Train FPS | Total Params |
|---|---:|---:|---:|---:|---:|---:|
| SEAC | | | | | | |
| SEAC-GRU | | | | | | |
| Shared-A2C | | | | | | |
| Shared-A2C-GRU | | | | | | |

The next research architecture should be chosen from evidence in this table.

---

## 15. Decision Rules

If Shared-A2C-GRU is strongest:
- use one shared recurrent policy as the new backbone
- then test whether redesigned Population-SEAC improves it

If SEAC-GRU is strongest:
- SEAC experience sharing remains valuable
- focus on scalable recurrent SEAC

If Shared-A2C and Shared-A2C-GRU are similar:
- memory may not matter at current congestion
- increase congestion before increasing architecture complexity

If SEAC and SEAC-GRU are similar:
- memory is not solving the main SEAC limitation
- focus on sharing/generalization instead

If all four perform poorly:
- diagnose reward, observation sufficiency, task difficulty, action distribution, and learning curves before adding complexity

---

## Immediate Next Coding Target

Implement:

\[
oxed{	ext{Shared-A2C}}
\]

first.

It is the smallest conceptual change and gives the essential shared-policy control.

Then:

\[
oxed{	ext{Shared-A2C-GRU}}
\]

because GRU support already exists in the policy.

Finally implement the recurrent-state correction and run:

\[
oxed{	ext{SEAC-GRU}}
\]

for the full fair 2×2 comparison.
