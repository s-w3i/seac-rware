#!/usr/bin/env python3
"""Inspect bounded CCPD rollouts from a saved MAPPO-GRU, without training it."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'seac/seac'))

from shared_ccpd import DETECTOR_VERSION, select_samples, event_records
from shared_envs import SharedEnvs, STATE_SCHEMA
from shared_ppo import SharedPPO
from train_shared import DEFAULTS, append_json, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rollouts', type=int, default=4, choices=range(1, 9))
    parser.add_argument('--seed', type=int, default=1000)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved['config']['method'] != 'mappo' or not saved['architecture']['recurrent'] or 'critic' not in saved:
        parser.error('Audit requires a full MAPPO-GRU checkpoint (best.pt or last.pt)')
    if saved['state_schema'] != STATE_SCHEMA:
        parser.error('Checkpoint uses an incompatible centralized-state schema')
    config = dict(DEFAULTS, **saved['config'])
    config.update(ccpd_mode='successful', seed=args.seed, device='cpu')
    validate(config)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    envs = SharedEnvs(config['env_name'], config['num_envs'], args.seed,
                      config['time_limit'], coordination_trace=True)
    try:
        a = saved['architecture']
        learner = SharedPPO(a['obs_size'], a['actions'], a['state_size'], config, torch.device('cpu'))
        learner.actor.load_state_dict(saved['actor'])
        learner.critic.load_state_dict(saved['critic'])
        args.output.mkdir(parents=True, exist_ok=False)
        totals = dict(events=0, successful=0, failed=0, censored=0, selected=0)
        for rollout in range(args.rollouts):
            data, _, _ = learner.collect(envs)
            _, metrics, events = select_samples(data, config, rollout)
            append_json(args.output / 'metrics.jsonl', dict(rollout=rollout, **metrics))
            for event in events:
                append_json(args.output / 'events.jsonl', dict(rollout=rollout, **event))
            if rollout < 4:
                for record in event_records(data, events):
                    append_json(args.output / 'traces.jsonl', dict(rollout=rollout, **record))
            totals['events'] += len(events)
            for key in ('successful', 'failed', 'censored'):
                totals[key] += metrics[f'ccpd_{key}_events']
            totals['selected'] += metrics['ccpd_selected_samples']
        # State-dict equality includes all learned parameters; there were no updates.
        for name in ('actor', 'critic'):
            assert all(torch.equal(tensor.cpu(), saved[name][key])
                       for key, tensor in getattr(learner, name).state_dict().items())
        summary = dict(checkpoint=str(args.checkpoint.resolve()), detector_version=DETECTOR_VERSION,
                       config=config, rollouts=args.rollouts, env_steps=args.rollouts * config['num_envs'] * config['rollout_steps'],
                       parameters_unchanged=True, **totals)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(totals))
    finally:
        envs.close()


if __name__ == '__main__':
    main()
