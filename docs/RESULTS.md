# Seed-0 results

These are **checkpoint-selection validation results** on the single adopted map: ten scenarios (100000–100009), three stochastic action replicates, 5,000 continuous steps. They are not 30 independent scenarios or an untouched test set. Incomplete tasks are not included in completed-cycle duration quantiles. All reported checkpoints have zero fleet completion-gap failures; individual failures are shown separately.

## Selected checkpoints

| Metric | 40 MAPPO | 40 Comm + CCPD | 50 MAPPO | 50 Comm + CCPD |
|---|---:|---:|---:|---:|
| Update | 343 | 196 | 196 | 49 |
| Cycles / 1,000 joint steps | 380.77 | 384.41 | 464.53 | 473.22 |
| Conflicts / 1,000 navigation decisions | 17.43 | 15.90 | 21.04 | 23.71 |
| Completed-cycle p95 duration (steps) | 163 | 162 | 167 | 164 |
| Completed-cycle p99 duration (steps) | 190 | 189 | 196 | 194 |
| Maximum task age anywhere (steps) | 442 | 463 | 496 | 343 |
| Individual progress-failure episodes | 0/30 | 0/30 | 0/30 | 0/30 |

Combined-model throughput gains are 0.96% at 40 agents and 1.87% at 50 agents. Conflicts per navigation decision fall 8.77% at 40 but rise 12.65% at 50. Neither gain meets the predeclared 3% throughput threshold; there is no paired interval across training seeds because only seed 0 completed. These results do not establish an overall winner or the isolated causal effect of CCPD.

## Final checkpoints (update 683)

| Fleet | Method | Cycles / 1,000 steps | Individual progress-failure episodes | Maximum task age |
|---|---|---:|---:|---:|
| 40 | MAPPO | 347.98 | 0/30 | 341 |
| 40 | Comm + CCPD | 371.43 | 1/30 | 578 |
| 50 | MAPPO | 404.68 | 4/30 | 829 |
| 50 | Comm + CCPD | 429.11 | 0/30 | 384 |

The 500-step rule is an operational screen. Zero observed failures does not prove deadlock freedom. The selected checkpoints are the primary pretrained models; final checkpoints are retained to expose late-training regression, not hidden.

## Evidence

Run `python scripts/pretrained.py --verify --report` to verify checksums and recompute throughputs/failure counts from the original per-episode logs. `evidence/<method>/n<fleet>/validation.jsonl.gz` retains all validation checkpoints; `metrics.jsonl.gz` retains update diagnostics. Original protocol and runtime-repair records retain their historical paths and source hashes. `pretrained/manifest.json` maps the release files to checksums and selected updates.

At n50 the selected-to-final training windows show increasing action entropy and more waiting/turning, while evaluation reward also declines. This motivates a separately declared adaptation-stability experiment; it does not establish a causal explanation. No such follow-up training is included here.
