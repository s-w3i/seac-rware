# CCPD v0 implementation validation

The implementation preserves the existing five-robot MAPPO-GRU architecture.
Full 20M-step training and comparisons have not been launched.

## Checks completed

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest seac/tests robotic-warehouse/tests -q`: **130 passed**, one existing Gymnasium list-reward warning.
- Three-update comparison against the saved pre-CCPD `shared_ppo.py`: exact actor/critic parameter, observation and Torch RNG equality with CCPD disabled.
- Automated zero-coefficient parity, empty-active-selection parity, raw-advantage/mask/cap checks, recurrent auxiliary loss, checkpoint/resume and actor export checks passed.
- Four CPU modes completed 4,096 environment steps each; all diagnostics finite and both networks updated.
- GPU CCPD completed 8,192 environment steps / four updates. All diagnostics finite; 41 selected actions produced nonzero auxiliary losses. Full checkpoint and actor export outputs matched after CPU reload.

| CPU mode | Events | Successful events | Selected actions |
|---|---:|---:|---:|
| off | disabled | disabled | 0 |
| random | 197 | 31 | 48 |
| all_conflict | 187 | 27 | 23 |
| successful | 188 | 20 | 15 |

The policies diverge after updates, so these counts need not match across runs.
The control selection test uses the same rollout to verify equal sample counts
and weight multisets. Smoke throughput does not establish learned performance.

CPU artifacts are in `cpu/<mode>/`; GPU artifacts are in
`gpu/mappo_gru_ccpd_routing/seed_0/gpu_smoke/train/`. The GPU smoke took 25.82
seconds within the trainer and allocated a peak 131,626,496 bytes in PyTorch
(125.53 MiB), excluding CUDA context. Other validation jobs were running, so
these timings are not an isolated speed benchmark.

## Read-only event audit

Input: `results/shared_baselines/mappo_gru_routing/seed_0/four_models_20m/train/best.pt`.
Four frozen-policy rollouts, eight environments, 256 steps each, reset seeds
1000–1007. Actor/critic parameters were verified unchanged. The audit's critic
supplies the existing raw team advantage; no training occurred.

343 events: **263 successful, 47 failed, 33 censored**. Failed reasons were
35 insufficient-progress, six late-conflict and six low-quality events.
407 actions passed the success, advantage, decision, denied-action and NOOP filters.
Full labels are in `trained_actor_audit/events.jsonl`; bounded step records are
in `trained_actor_audit/traces.jsonl`; configuration and totals are in its summary.

Manually inspected examples:

- Rollout 0, environment 0, robot 2, steps 0–11: a denied forward request,
  a turn, two successful translations, then five cells of confirmation progress.
  Quality 3.6, accepted. Automatic pickup measured distance 0→0; the next
  target's distance 11 did not create an artificial jump.
- Rollout 1, environment 4, robot 4, steps 8–19: movement resumed and delivery
  occurred, but subsequent return navigation made net progress −3. Rejected.
- Rollout 0, environment 3, robot 1, steps 2–13: one cell of confirmation
  progress followed by another blocked request on the last confirmation step.
  Rejected for late conflict.
- Rollout 0, environment 2, robot 2, steps 10–22: one cell of confirmation
  progress after a five-step event and one additional conflict. Quality 0,
  rejected by the strict positive-quality threshold.

Scripted tests separately cover persistent waiting/timeouts, wall rejection,
invalid distances, censored windows and exact episode-boundary completion.
The thresholds were not weakened to produce successful examples.

## Next experiment

Use the documented CCPD launcher command in the root README for fresh seeds
0, 1 and 2 at 20M steps, then evaluate best and final checkpoints alongside
the existing MAPPO-GRU and Shared-PPO-GRU on seeds 2000–2049. Run the auxiliary
controls afterward. A successful implementation does not establish a throughput gain.
