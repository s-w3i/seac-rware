#!/usr/bin/env python3
"""Train CCPD seeds 0/1/2 concurrently on physical GPUs 0/1/1."""
import argparse
from pathlib import Path
import shlex

from run_shared_baselines import ROOT, plan_jobs, prepare_jobs, run_wave


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-env-steps', type=int, default=20000000)
    parser.add_argument('--attempt', default='ccpd_v0_20m')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ccpd')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-busy', action='store_true')
    parser.set_defaults(models=['mappo_gru_ccpd_routing'], gpus=['0', '1'], jobs_per_gpu=1,
                        python=ROOT / '.venv/bin/python', env_name=None,
                        num_envs=None, rollout_steps=None, eval_steps=None)
    args = parser.parse_args(argv)
    if args.num_env_steps < 1:
        parser.error('--num-env-steps must be positive')
    if Path(args.attempt).name != args.attempt or args.attempt in ('', '.', '..'):
        parser.error('--attempt must be a single directory name')
    jobs = []
    for seed, gpu in ((0, '0'), (1, '1'), (2, '1')):
        args.seed = seed
        job, = plan_jobs(args)
        job['physical_gpu'] = gpu
        job['command'].append('ccpd_trace_events=True')
        jobs.append(job)
        print(f"seed={seed} physical_gpu={gpu} log={Path(job['run_dir']) / 'train.log'}", flush=True)
        if args.dry_run:
            print(f'CUDA_VISIBLE_DEVICES=<UUID-of-{gpu}> ' + shlex.join(job['command']))
    if args.dry_run:
        return 0
    prepare_jobs(jobs, args.gpus, args.allow_busy, args.python)
    print('Starting all three seeds concurrently. Ctrl+C stops these training processes.', flush=True)
    return run_wave(jobs)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
