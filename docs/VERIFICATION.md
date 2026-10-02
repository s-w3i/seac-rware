# Release verification — 2026-10-02

- Checksums verified for the runtime Python sources, all published model/source checkpoints, map and evidence.
- Regression suite in the relocated release checkout: 188 passed, 3 CUDA-dependent skips (CPU sandbox). This included the nine pretrained-artifact/export tests. A further test of seed-0 wrapper child commands and the automatic-campaign guard passed separately.
- Both methods completed disposable CPU smoke training through 1 and 50 agents, two updates each, with original Stage-1 sources. No production training was launched.
- All eight actor-only exports (selected/final × 40/50 agents × two methods) matched their full checkpoint weights exactly and produced identical actions/log probabilities during simulator checks.
- One complete original validation episode was re-run for **each** selected 50-agent model: scenario 100000, replicate 0, 5,000 steps. Every original episode field matched exactly except wall-clock inference timing. MAPPO completed 2,313 cycles; communication + CCPD completed 2,353. See `validation_replay_check.json`.
- Video command produced a 21-frame, 50-agent MP4 with the combined policy under Xvfb.
- Report command recomputed the selected/final throughputs and progress-failure counts from original compressed episode evidence.

The complete 30-episode validation suite was not re-run for every checkpoint during packaging; the original logs are preserved. Fresh installations without the local historical CCPD archive additionally skip its archive-only regression check. GPU-specific checks are included for execution on a CUDA host. Neither fresh holdout evaluation nor additional production training was run.
