#!/usr/bin/env python3
"""Launch independent shared PPO experiments, one or two jobs per GPU."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('shared_ppo_routing', 'mappo_routing', 'shared_ppo_gru_routing', 'mappo_gru_routing')


def plan_jobs(args):
    jobs = []
    counts = [0, 0]
    for model in MODELS:
        if model not in args.models:
            continue
        slot = int(model.startswith('mappo'))
        wave = counts[slot] // args.jobs_per_gpu
        counts[slot] += 1
        run_dir = (args.output / model / f'seed_{args.seed}' / args.attempt).resolve()
        cmd = [str(args.python), '-u', str(ROOT / 'seac/seac/train_shared.py'), 'with', model,
               f'seed={args.seed}', 'device=cuda:0', f'num_env_steps={args.num_env_steps}',
               f'run_dir={run_dir / "train"}']
        for key in ('env_name', 'num_envs', 'rollout_steps', 'eval_steps'):
            value = getattr(args, key)
            if value is not None:
                cmd.append(f'{key}={value}')
        jobs.append(dict(model=model, seed=args.seed, physical_gpu=args.gpus[slot],
                         wave=wave, run_dir=str(run_dir), command=cmd, cwd=str(ROOT / 'seac/seac')))
    return jobs


def resolve_gpus(selectors, allow_busy):
    listing = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    available = dict(line.replace(' ', '').split(',') for line in listing.strip().splitlines())
    if len(set(selectors)) != 2 or any(s not in available for s in selectors):
        raise ValueError(f'Choose two distinct physical GPU indices from {list(available)}')
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    if inherited is not None:
        allowed = {available.get(s.strip(), s.strip()) for s in inherited.split(',')}
        if not all(available[s] in allowed for s in selectors):
            raise ValueError('Requested physical GPUs conflict with inherited CUDA_VISIBLE_DEVICES')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                         '--format=csv,noheader'], text=True)
    busy = [line for line in processes.splitlines() if line.split(',')[0].strip() in {available[s] for s in selectors}]
    if busy:
        print('Existing GPU processes:\n' + '\n'.join(busy), file=sys.stderr)
        if not allow_busy:
            raise RuntimeError('Selected GPU is busy; use --allow-busy to intentionally share it')
    return {s: available[s] for s in selectors}


def run_wave(jobs):
    """Run prepared jobs concurrently; terminate/reap only owned children."""
    running = []
    failed = False
    previous = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Launcher received signal {signum}')

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        for job in jobs:
            directory = Path(job['run_dir'])
            log = (directory / 'train.log').open('x')
            record = dict(job, started_at=time.time())
            try:
                child_env = os.environ.copy()
                child_env.update(job['environment'])
                process = subprocess.Popen(job['command'], cwd=job['cwd'], env=child_env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException as error:
                log.close()
                record.update(error=str(error), exit_status=-1, finished_at=time.time())
                (directory / 'launch.json').write_text(json.dumps(record, indent=2))
                raise
            record['pid'] = process.pid
            running.append((process, log, directory, record))
            (directory / 'launch.json').write_text(json.dumps(record, indent=2))
        while any(p.poll() is None for p, _, _, _ in running):
            if any(p.poll() not in (None, 0) for p, _, _, _ in running):
                failed = True
                break
            time.sleep(.1)
        failed = failed or any(p.poll() not in (None, 0) for p, _, _, _ in running)
    finally:
        # Ignore repeated interrupts during bounded cleanup, then restore handlers.
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        for process, _, _, _ in running:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 5
        for process, log, directory, record in running:
            try:
                process.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.close()
            record.update(exit_status=process.returncode, finished_at=time.time())
            (directory / 'launch.json').write_text(json.dumps(record, indent=2))
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--gpus', nargs=2, default=['0', '1'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--num-env-steps', type=int, default=2000000)
    parser.add_argument('--jobs-per-gpu', type=int, choices=(1, 2), default=1)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/shared_baselines')
    parser.add_argument('--attempt', default='attempt_1')
    parser.add_argument('--python', type=Path, default=ROOT / '.venv/bin/python')
    parser.add_argument('--env-name', default=None)
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--rollout-steps', type=int)
    parser.add_argument('--eval-steps', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-busy', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.gpus)) != 2 or len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        parser.error('GPU indices, models and seeds must not contain duplicates')
    if any(seed < 0 for seed in args.seeds) or any(v is not None and v < 1 for v in
            (args.num_env_steps, args.num_envs, args.rollout_steps, args.eval_steps)):
        parser.error('Seeds must be nonnegative and step/environment counts positive')
    if Path(args.attempt).name != args.attempt or args.attempt in ('', '.', '..'):
        parser.error('--attempt must be a single directory name')
    args.python = args.python.absolute()
    jobs = []
    for seed in args.seeds:
        args.seed = seed
        jobs.extend(plan_jobs(args))
    if args.dry_run:
        for job in jobs:
            print(f"seed={job['seed']} wave={job['wave'] + 1} physical_gpu={job['physical_gpu']} "
                  f"CUDA_VISIBLE_DEVICES=<UUID-of-{job['physical_gpu']}> " + shlex.join(job['command']))
        return 0
    existing = [j['run_dir'] for j in jobs if Path(j['run_dir']).exists()]
    if existing:
        raise FileExistsError('Existing attempts: ' + ', '.join(existing))
    gpu_ids = resolve_gpus(args.gpus, args.allow_busy)
    for gpu, gpu_uuid in gpu_ids.items():
        probe_env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid)
        subprocess.run([str(args.python), '-c',
                        'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; torch.zeros(1,device="cuda:0")'],
                       env=probe_env, check=True)
    for job in jobs:
        Path(job['run_dir']).mkdir(parents=True, exist_ok=False)
        job['environment'] = dict(CUDA_VISIBLE_DEVICES=gpu_ids[job['physical_gpu']],
                                  PHYSICAL_GPU_ID=job['physical_gpu'], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for seed in args.seeds:
        for wave in sorted({j['wave'] for j in jobs if j['seed'] == seed}):
            selected = [j for j in jobs if j['seed'] == seed and j['wave'] == wave]
            if run_wave(selected):
                return 1
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
