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
