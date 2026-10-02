#!/usr/bin/env python3
"""Validate by default. Only --train launches the 240M-step research campaign."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/home/utar/seac-rware')
PYTHON = ORIGINAL / '.venv/bin/python'
sys.path[:0] = [str(ROOT / 'robotic-warehouse'), str(ROOT / 'seac/seac')]

import torch
from navigation_envs import load_layouts, geometry_hash
from navigation_train import (CONDITIONS, configuration, verify_training, evaluate, load_checkpoint,
                              sha, source_hashes, write_json)
from run_shared_baselines import free_gpus, run_wave


def child_environment():
    return dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT/'robotic-warehouse'), str(ROOT/'seac/seac'), str(ROOT/'scripts')]),
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def signature(manifest):
    values = source_hashes()
    for folder in ('seac/tests', 'robotic-warehouse/tests'):
        values.update({str(p.relative_to(ROOT)): sha(p) for p in (ROOT/folder).glob('*.py')})
    values['manifest'] = sha(manifest)
    values['layouts'] = [r['sha256'] for r in load_layouts(manifest)]
    values['runtime'] = {package: importlib.metadata.version(package) for package in ('torch', 'numpy', 'gymnasium', 'networkx')}
    return digest(values)


def receipt_valid(path, inputs):
    marker = path.with_name(path.name+'.done.json')
    partial = path.with_name(path.name+'.partial')
    if partial.exists():
        raise ValueError(f'Interrupted artifact; inspect and explicitly recover {partial}')
    if not path.exists() and not marker.exists():
        return False
    if not path.exists() or not marker.exists() or json.loads(marker.read_text()) != dict(inputs=inputs, sha256=sha(path)):
        raise ValueError(f'Incomplete/incompatible artifact: {path}')
    return True


def run_validation(output, manifest, log):
    code = signature(manifest)
    directory = output/'validation'/code
    result = directory/'result.json'
    if result.exists():
        value = json.loads(result.read_text())
        if not value.get('passed') or value['signature'] != code or any(sha(directory/name) != checksum for name, checksum in value['artifacts'].items()):
            raise ValueError(f'Invalid validation result {result}')
        log('Reusing verified validation: '+str(result))
        return result
    if directory.exists():
        raise ValueError(f'Incomplete validation; inspect {directory}')
    directory.mkdir(parents=True)
    log('Running regression tests: '+str(directory/'tests.log'))
    with (directory/'tests.log').open('x') as handle:
        result_code = subprocess.run([str(PYTHON), '-m', 'pytest', 'seac/tests', 'robotic-warehouse/tests', '-q'],
                                     cwd=ROOT, env=child_environment(), stdout=handle, stderr=subprocess.STDOUT).returncode
    if result_code:
        raise RuntimeError(f"Regression tests failed; inspect {directory/'tests.log'}")
    for condition in CONDITIONS:
        config = configuration(manifest, condition, 0, budget=4096)
        config_file = directory/(condition+'.json')
        write_json(config_file, config)
        log('4096-step smoke: '+condition)
        with (directory/(condition+'.log')).open('x') as handle:
            result_code = subprocess.run([str(PYTHON), '-u', str(ROOT/'seac/seac/navigation_train.py'), '--config', str(config_file),
                                         '--output', str(directory/condition), '--smoke'], cwd=ROOT, env=child_environment(),
                                        stdout=handle, stderr=subprocess.STDOUT).returncode
        if result_code:
            raise RuntimeError(f"Smoke failed; inspect {directory/(condition+'.log')}")
        verify_training(directory/condition, config)
        rows = [json.loads(l) for l in (directory/condition/'metrics.jsonl').read_text().splitlines()]
        if config['ccpd_mode'] != 'off' and sum(r['ccpd_selected_samples'] for r in rows) <= 0:
            raise ValueError('Smoke selected no auxiliary actions')
        summary = json.loads((directory/condition/'summary.json').read_text())
        if not summary['actor_changed'] or not summary['critic_changed']:
            raise ValueError('Smoke failed to update parameters')
    if signature(manifest) != code:
        raise ValueError('Code changed during validation')
    artifacts = {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*') if p.is_file()}
    write_json(result, dict(passed=True, signature=code, smoke_steps_per_condition=4096, artifacts=artifacts))
    log('Validation complete: '+str(result))
    return result


def train_campaign(output, manifest, log, verify):
    pending = []
    for seed in range(3):
        for condition in CONDITIONS:
            config = configuration(manifest, condition, seed, device='cuda:0')
            directory = output/'training'/condition/f'seed_{seed}'
            if directory.exists():
                verify_training(directory/'train', config)
                launch = json.loads((directory/'launch.json').read_text())
                if launch.get('exit_status') != 0:
                    raise ValueError(f'Training subprocess did not complete: {directory}')
                log(f'Reusing completed {condition}, seed {seed}')
            else:
                pending.append((condition, seed, config, directory))
    while pending:
        verify()
        available = dict(free_gpus())
        wave = [job for job in pending if job[1] == pending[0][1]]
        assignment = {condition: ('0' if condition == 'local' else '1') for condition in CONDITIONS}
        required = {assignment[job[0]] for job in wave}
        if not required <= available.keys():
            log('Waiting for assigned GPUs '+str(sorted(required))+'; unrelated jobs remain untouched')
            time.sleep(30)
            continue
        jobs = []
        for condition, seed, config, directory in wave:
            gpu = assignment[condition]
            uuid = available[gpu]
            config_file = output/'configs'/f'{condition}_{seed}.json'
            write_json(config_file, config)
            jobs.append(dict(model=condition, seed=seed, physical_gpu=gpu, run_dir=str(directory),
                             command=[str(PYTHON), '-u', str(ROOT/'seac/seac/navigation_train.py'),
                                      '--config', str(config_file), '--output', str(directory/'train')], cwd=str(ROOT),
                             environment=dict(child_environment(), CUDA_VISIBLE_DEVICES=uuid, PHYSICAL_GPU_ID=gpu)))
        if not {j['environment']['CUDA_VISIBLE_DEVICES'] for j in jobs} <= {uuid for _, uuid in free_gpus()}:
            continue
        for job in jobs:
            Path(job['run_dir']).mkdir(parents=True, exist_ok=False)
            log(f"Starting {job['model']} seed {job['seed']} GPU {job['physical_gpu']}")
        if run_wave(jobs):
            raise RuntimeError('Training failed: inspect training/*/seed_*/train.log and launch.json')
        for _, _, config, directory in wave:
            verify_training(directory/'train', config)
        pending = [job for job in pending if job not in wave]


def suites(communication, kind='last'):
    fast = os.environ.get('DECENTRALIZED_FAST_EVAL') == '1'
    base = ([('reference_short', 'reference', 500, 10),
             ('reference_long', 'reference', 10000, 10)] if fast else
            [('reference_short', 'reference', 500, 50),
             ('reference_long', 'reference', 10000, 50)])
    if kind == 'last':
        base += ([('changed_tasks', 'changed_tasks', 10000, 10),
                  ('changed_starts', 'changed_starts', 10000, 10)] if fast else
                 [('changed_tasks', 'changed_tasks', 10000, 20),
                  ('changed_starts', 'changed_starts', 10000, 20)])
    replicates = 1 if fast else 3
    result = [dict(suite=name, profile=profile, steps=steps, count=count, replicate=r, deterministic=False, loss=.1)
              for name, profile, steps, count in base for r in range(replicates)]
    if kind == 'last':
        result += [dict(suite=name+'_deterministic', profile=profile, steps=steps, count=count, replicate=0, deterministic=True, loss=.1)
                   for name, profile, steps, count in base[:2]]
        if communication and not fast:
            result += [dict(suite=f'loss_{loss}_long', profile='reference', steps=10000, count=50, replicate=r, deterministic=False, loss=loss)
                       for loss in (0., .3, 1.) for r in range(3)]
    return result


def evaluation_campaign(output, manifest, log, verify, checkpoint_root=None):
    checkpoint_root = Path(checkpoint_root) if checkpoint_root else output
    layouts = load_layouts(manifest)
    if len(layouts) != 1:
        raise ValueError('This campaign evaluates one layout; use the layout/evaluation APIs for a separately declared transfer experiment')
    layout = layouts[0]
    for condition, (communication, _) in CONDITIONS.items():
        for seed in range(3):
            for kind in ('last', 'best'):
                checkpoint = checkpoint_root/'training'/condition/f'seed_{seed}'/'train'/f'{kind}.pt'
                actor, saved = load_checkpoint(checkpoint)
                checkpoint_hash = sha(checkpoint)
                for spec in suites(communication, kind):
                    verify()
                    path = output/'evaluation'/condition/f'seed_{seed}'/f"{kind}_{spec['suite']}_rng_{spec['replicate']}.jsonl"
                    inputs = dict(checkpoint_sha256=checkpoint_hash, layout_sha256=layout['sha256'], protocol=spec, sources=source_hashes())
                    if not receipt_valid(path, inputs):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with path.with_name(path.name+'.partial').open('x') as handle:
                            rows = evaluate(actor, layout, range(2000, 2000+spec['count']), spec['steps'],
                                            replicate=spec['replicate'], deterministic=spec['deterministic'],
                                            profile=spec['profile'], loss=spec['loss'])
                            for row in rows:
                                row.update(method=condition, training_seed=seed, checkpoint_kind=kind, suite=spec['suite'],
                                           checkpoint_sha256=checkpoint_hash, checkpoint_env_steps=saved['env_steps'])
                                handle.write(json.dumps(row, allow_nan=False)+'\n')
                        if sha(checkpoint) != checkpoint_hash or any(not torch.equal(v, saved['actor'][k]) for k, v in actor.state_dict().items()):
                            raise ValueError('Evaluation changed checkpoint parameters')
                        verify()
                        path.with_name(path.name+'.partial').rename(path)
                        write_json(path.with_name(path.name+'.done.json'), dict(inputs=inputs, sha256=sha(path)))
                    rows = [json.loads(l) for l in path.read_text().splitlines()]
                    if [r['episode_id'] for r in rows] != list(range(2000, 2000+spec['count'])) or any(r['steps'] != spec['steps'] for r in rows):
                        raise ValueError(f'Incomplete evaluation: {path}')
                    log(f"Evaluated {condition} seed {seed}, {kind}, {spec['suite']}, RNG {spec['replicate']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true', help='Explicitly launch full training after validation')
    parser.add_argument('--eval-only', action='store_true', help='Evaluate existing checkpoints without validation or training')
    parser.add_argument('--fast-eval', action='store_true', help='Use the reduced evaluation suite')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path, default=ORIGINAL/'results/decentralized_navigation')
    parser.add_argument('--layout-manifest', type=Path, default=ROOT/'assets/navigation_layouts.json')
    args = parser.parse_args(argv)
    if args.fast_eval:
        os.environ['DECENTRALIZED_FAST_EVAL'] = '1'
    torch.set_num_threads(1)
    import rware
    if Path(rware.__file__).resolve() != ROOT/'robotic-warehouse/rware/__init__.py':
        raise RuntimeError('Incorrect simulator import')
    layouts = load_layouts(args.layout_manifest)
    if args.train and (len(layouts) != 1 or layouts[0]['id'] != 'original' or layouts[0]['sha256'] != geometry_hash(rware.CUSTOM_5_ROBOT_LAYOUT)):
        raise ValueError('The declared campaign trains on the original layout only; multi-layout sampling is tested but not activated')
    if args.dry_run:
        print(json.dumps(dict(mode='training' if args.train else 'validation', training_runs=12 if args.train else 0,
                              budget=240_000_000 if args.train else 0, smoke_steps=4096, conditions=list(CONDITIONS),
                              gpu_assignment={c: ('0' if c == 'local' else '1') for c in CONDITIONS},
                              output=str(args.output), layout_hashes=[r['sha256'] for r in layouts]), indent=2))
        return 0
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if args.eval_only and args.train:
        parser.error('--eval-only and --train are mutually exclusive')
    with (output/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                            handlers=[logging.FileHandler(output/'pipeline.log'), logging.StreamHandler()])
        log = logging.info
        def status(stage, **extra):
            write_json(output/'status.json', dict(stage=stage, updated_at=time.time(), pid=os.getpid(), **extra))
        try:
            if args.eval_only:
                status('evaluation')
                source_output = ORIGINAL/'results/decentralized_navigation'
                manifest = args.layout_manifest
                evaluation_campaign(output, manifest, log, lambda: None, checkpoint_root=source_output)
                status('complete')
                print(f'Evaluation complete: {output}')
                return 0
            status('validation')
            validation = run_validation(output, args.layout_manifest, log)
            if not args.train:
                status('validated', validation=str(validation), full_training_launched=False)
                print('Validated. Full training was not launched. '+str(validation))
                return 0
            protocol = dict(sources=source_hashes(), manifest_sha256=sha(args.layout_manifest),
                            layout_hashes=[r['sha256'] for r in layouts], validation_signature=signature(args.layout_manifest),
                            budget=20_000_000, seeds=[0, 1, 2], conditions=CONDITIONS)
            protocol = json.loads(json.dumps(protocol))
            freeze = output/'protocol.json'
            if freeze.exists() and json.loads(freeze.read_text()) != protocol:
                artifacts = list((output/'training').glob('*/seed_*')) + list((output/'evaluation').glob('**/*.jsonl'))
                if artifacts:
                    raise ValueError('Existing campaign uses a different protocol; do not overwrite it')
                write_json(freeze, protocol)
            if not freeze.exists():
                write_json(freeze, protocol)
            def verify():
                if source_hashes() != protocol['sources'] or sha(args.layout_manifest) != protocol['manifest_sha256']:
                    raise ValueError('Frozen campaign sources/manifest changed')
                load_layouts(args.layout_manifest)
            (output/'configs').mkdir(exist_ok=True)
            status('training'); train_campaign(output, args.layout_manifest, log, verify)
            status('evaluation'); evaluation_campaign(output, args.layout_manifest, log, verify)
            status('reporting')
            from report_navigation import generate_report
            generate_report(output)
            status('complete')
            print(f"Reports: {output/'comparison.md'}; {output/'next_steps.md'}; {output/'summary.json'}")
            return 0
        except BaseException as exc:
            status('failed', error=str(exc), log=str(output/'pipeline.log'))
            raise


if __name__ == '__main__':
    raise SystemExit(main())
