#!/usr/bin/env python3
"""Run the CCPD investigation and matched 80M-step pilot with one command.

Default: wait for the old campaign, analyze, replay, audit, train, evaluate,
report. --check runs only tests and a CPU smoke run. --dry-run creates nothing.
"""
import argparse
import fcntl
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from investigate_ccpd import (ROOT, ORIGINAL, METHODS, MODES, SCENARIOS, assert_isolation,
                             sha, read_rows, original_train, cached, save_artifact, finish,
                             replay_failures, audit_selections, write_json)
import numpy as np
import torch
from evaluate_shared import load_actor, evaluate_actor, checkpoint_metadata
from run_shared_baselines import run_wave
from validate_ccpd import layouts

PYTHON = ORIGINAL / '.venv/bin/python'
BUDGET = 10_000_000
STAGES = ('existing_evidence', 'failure_replay', 'selection_audit', 'pilot_training', 'pilot_evaluation', 'report')


def child_environment():
    return dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / 'robotic-warehouse'), str(ROOT / 'seac/seac'))),
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')


def source_hashes():
    paths = [*list((ROOT / 'seac/seac').glob('*.py')), *list((ROOT / 'seac/seac/configs').glob('*.yaml')),
             *list((ROOT / 'robotic-warehouse/rware').glob('*.py')),
             *(ROOT / 'scripts' / n for n in ('run_ccpd_investigation.py', 'investigate_ccpd.py',
               'report_ccpd_investigation.py', 'run_shared_baselines.py', 'validate_ccpd.py', 'summarize_ccpd_validation.py'))]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def validation_signature():
    values = source_hashes()
    for folder in ('seac/tests', 'robotic-warehouse/tests'):
        values.update({str(p.relative_to(ROOT)): sha(p) for p in (ROOT / folder).glob('*.py')})
    return sha_bytes(values)


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prerequisite_state(prior):
    status_path, lock_path = prior / 'status.json', prior / 'pipeline.lock'
    if not status_path.exists() or not lock_path.exists():
        raise RuntimeError(f'Missing prerequisite campaign: {prior}')
    with lock_path.open('r') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            active = False
        except BlockingIOError:
            active = True
        state = json.loads(status_path.read_text())
    if state.get('stage') == 'failed':
        raise RuntimeError(f'Prerequisite failed: {state.get("error")}; inspect {prior / "pipeline.log"}')
    if active:
        return 'active'
    if state.get('stage') != 'complete':
        raise RuntimeError(f'Prerequisite stopped incomplete ({state.get("stage")}): {status_path}')
    if not (prior / 'comparison.md').exists() or not json.loads((prior / 'summary.json').read_text()).get('complete'):
        raise RuntimeError(f'Prerequisite report is missing or incomplete: {prior}')
    return 'complete'


def wait_prerequisite(prior, log):
    while prerequisite_state(prior) != 'complete':
        log(f'Waiting for current campaign: {prior / "status.json"}')
        time.sleep(30)


def verify_prior(prior):
    hashes = {}
    # The old campaign's source freeze remains authoritative and is read only.
    for relative, digest in json.loads((prior / 'protocol_sources.json').read_text()).items():
        if sha(ORIGINAL / relative) != digest:
            raise ValueError(f'Original campaign source changed: {relative}')
    for method in METHODS:
        for seed in range(3):
            directory = original_train(prior, method, seed)
            for name in ('summary.json', 'provenance.json', 'metrics.jsonl', 'evaluation.jsonl', 'last.pt'):
                path = directory / name
                hashes[str(path)] = sha(path)
            summary = json.loads((directory / 'summary.json').read_text())
            if not 20_000_000 <= summary['env_steps'] < 20_002_048:
                raise ValueError(f'Incomplete original training: {directory}')
            for kind in ('last', 'best'):
                for scenario in SCENARIOS if kind == 'last' else ('reference',):
                    path = prior / 'evaluation' / method / f'seed_{seed}' / f'{kind}_{scenario}.jsonl'
                    rows = read_rows(path)
                    if len(rows) != 50 or [r['seed'] for r in rows] != list(range(2000, 2050)):
                        raise ValueError(f'Incomplete original evaluation: {path}')
                    checkpoint = directory / f'{kind}.pt'
                    digest = sha(checkpoint)
                    if any(r['method'] != method or r['training_seed'] != seed or r['scenario'] != scenario
                           or r['checkpoint_kind'] != kind or r['checkpoint_sha256'] != digest
                           or not r['deterministic'] or r['requested_steps'] != (10000 if scenario == 'long_run' else 500)
                           or r['continuous'] != (scenario == 'long_run') for r in rows):
                        raise ValueError(f'Incompatible original evaluation: {path}')
                    hashes[str(path)] = sha(path)
                    hashes[str(checkpoint)] = digest
    hashes[str(prior / 'summary.json')] = sha(prior / 'summary.json')
    return hashes


def run_checks(output, log):
    signature = validation_signature()
    result = output / 'validation' / signature / 'result.json'
    if result.exists():
        previous = json.loads(result.read_text())
        if (not previous.get('passed') or previous.get('signature') != signature
                or not previous.get('artifacts')
                or any(not Path(p).is_file() or sha(p) != digest for p, digest in previous['artifacts'].items())):
            raise ValueError(f'Validation artifacts changed or incomplete: {result}')
        log(f'Reusing validated code: {signature[:12]}')
        return
    result.parent.mkdir(parents=True, exist_ok=True)
    test_log = result.parent / 'tests.log'
    log('Running regression tests in the isolated checkout')
    with test_log.open('x') as handle:
        subprocess.run([str(PYTHON), '-m', 'pytest', 'seac/tests', 'robotic-warehouse/tests', '-q'],
                       cwd=ROOT, env=child_environment(), stdout=handle, stderr=subprocess.STDOUT, check=True)
    smoke = result.parent / 'smoke'
    with (result.parent / 'smoke.log').open('x') as handle:
        subprocess.run([str(PYTHON), '-u', str(ROOT / 'seac/seac/train_shared.py'), 'with', 'mappo_gru_ccpd_routing',
                        'ccpd_mode=random_action_matched', 'device=cpu', 'num_env_steps=4096',
                        'ccpd_trace_events=True', f'run_dir={smoke}'], cwd=ROOT, env=child_environment(),
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    rows = [r for r in read_rows(smoke / 'metrics.jsonl') if r.get('kind') == 'learning']
    if (not rows or rows[-1]['env_steps'] != 4096
            or sum(r['ccpd_selected_samples'] for r in rows) <= 0
            or any(not math.isfinite(v) for r in rows for v in r.values() if isinstance(v, (int, float)))
            or any(r['ccpd_selected_noop_fraction'] > .25 + 1e-8 or r['ccpd_selected_fraction'] > .1 + 1e-8 for r in rows)):
        raise ValueError(f'Smoke diagnostics failed: {smoke}')
    write_json(result, dict(passed=True, signature=signature, tests_log=str(test_log), smoke_steps=4096,
                            selected_actions=sum(r['ccpd_selected_samples'] for r in rows),
                            artifacts={str(p): sha(p) for p in (test_log, smoke / 'summary.json', smoke / 'metrics.jsonl', smoke / 'last.pt', smoke / 'provenance.json')},
                            simulator=str(ROOT / 'robotic-warehouse/rware/__init__.py')))
    log(f'Tests and 4096-step action-matched smoke passed: {result}')


def pilot_config(template, output, mode, seed):
    return dict(template, seed=seed, ccpd_mode=mode, num_env_steps=BUDGET, device='cuda:0', resume=None,
                run_dir=str(output / 'training' / mode / f'seed_{seed}' / 'train'))


def free_gpus():
    listing = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    available = dict(line.replace(' ', '').split(',') for line in listing.splitlines() if line.strip())
    restricted = os.environ.get('CUDA_VISIBLE_DEVICES')
    allowed = None if restricted is None else {available.get(s.strip(), s.strip()) for s in restricted.split(',')}
    permitted = {k: v for k, v in available.items() if allowed is None or v in allowed}
    if not permitted:
        raise RuntimeError('No CUDA GPU is permitted by the current environment')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    busy = {line.split(',')[0].strip() for line in processes.splitlines() if line.strip()}
    return [(index, uuid) for index, uuid in sorted(permitted.items(), key=lambda pair: int(pair[0])) if uuid not in busy]


def validate_training(directory, config, record=False):
    summary_path = directory / 'summary.json'
    if not summary_path.exists() or not (directory.parent / 'launch.json').exists():
        raise ValueError(f'Incomplete training requires explicit recovery: {directory}')
    if json.loads((directory.parent / 'launch.json').read_text()).get('exit_status') != 0:
        raise ValueError(f'Training did not exit successfully: {directory}')
    summary = json.loads(summary_path.read_text())
    expected = math.ceil(config['num_env_steps'] / (config['num_envs'] * config['rollout_steps'])) * config['num_envs'] * config['rollout_steps']
    if summary['env_steps'] != expected or json.loads((directory / 'provenance.json').read_text())['config'] != config:
        raise ValueError(f'Incompatible pilot training: {directory}')
    actor, saved = load_actor(directory / 'last.pt')
    if saved['config'] != config or saved['env_steps'] != expected or saved['architecture'] != dict(obs_size=199, actions=4, state_size=320, recurrent=True):
        raise ValueError(f'Invalid pilot checkpoint: {directory}')
    if any(not torch.isfinite(p).all() for p in actor.parameters()):
        raise ValueError(f'Nonfinite pilot actor: {directory}')
    for name in ('shared_ccpd.py', 'shared_ppo.py', 'shared_models.py', 'shared_envs.py', 'evaluate_shared.py', 'train_shared.py', 'shared_storage.py', 'warehouse.py'):
        current = ROOT / ('robotic-warehouse/rware' if name == 'warehouse.py' else 'seac/seac') / name
        if sha(directory / 'source' / name) != sha(current):
            raise ValueError(f'Pilot source changed: {directory / "source" / name}')
    result = dict(config=config, env_steps=expected,
                  artifacts={name: sha(directory / name) for name in ('last.pt', 'summary.json', 'metrics.jsonl', 'evaluation.jsonl', 'provenance.json')})
    marker = directory.parent / 'complete.json'
    inputs = dict(config=config)
    if not cached(marker, inputs):
        if not record:
            raise ValueError(f'Unverified training completion requires explicit recovery: {directory}')
        save_artifact(marker, result, inputs)
    if json.loads(marker.read_text()) != result:
        raise ValueError(f'Completed training artifacts changed: {directory}')
    return result


def train_pilot(prior, output, log, verify):
    template = json.loads((original_train(prior, 'ccpd', 0) / 'provenance.json').read_text())['config']
    pending = []
    for seed in range(2):
        for mode in MODES:
            config = pilot_config(template, output, mode, seed)
            directory = Path(config['run_dir'])
            if directory.parent.exists():
                validate_training(directory, config)
                log(f'Reusing completed pilot: {mode}, seed {seed}')
            else:
                pending.append((mode, seed, config))
    while pending:
        verify()
        free = free_gpus()
        if not free:
            log('Waiting for a free GPU; existing jobs will not be interrupted')
            time.sleep(30)
            continue
        jobs = []
        for (gpu, uuid), (mode, seed, config) in zip(free, pending):
            probe = dict(child_environment(), CUDA_VISIBLE_DEVICES=uuid)
            subprocess.run([str(PYTHON), '-c', 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; torch.zeros(1,device="cuda:0")'], env=probe, check=True)
            directory = Path(config['run_dir'])
            config_path = output / 'configs' / f'{mode}_{seed}.json'
            write_json(config_path, config)
            jobs.append(dict(model=mode, seed=seed, physical_gpu=gpu, run_dir=str(directory.parent),
                             command=[str(PYTHON), '-u', str(ROOT / 'seac/seac/train_shared.py'), 'with', str(config_path)],
                             cwd=str(ROOT / 'seac/seac'), environment=dict(probe, PHYSICAL_GPU_ID=gpu)))
        # Recheck occupancy immediately before starting the wave.
        if not {j['environment']['CUDA_VISIBLE_DEVICES'] for j in jobs} <= {uuid for _, uuid in free_gpus()}:
            continue
        for job in jobs:
            Path(job['run_dir']).mkdir(parents=True, exist_ok=False)
            log(f'Training {job["model"]}, seed {job["seed"]}, GPU {job["physical_gpu"]}')
        if run_wave(jobs):
            raise RuntimeError('Pilot training failed; see training/*/seed_*/train.log and launch.json')
        for _, _, config in pending[:len(jobs)]:
            validate_training(Path(config['run_dir']), config, record=True)
        pending = pending[len(jobs):]


def evaluation_protocol(scenario):
    count = 50 if scenario == 'reference' else 10 if scenario == 'long_run' else 20
    return list(range(2000, 2000 + count)), 10000 if scenario == 'long_run' else 500


def evaluate_pilot(output, log, verify):
    for mode in MODES:
        for seed in range(2):
            checkpoint = output / 'training' / mode / f'seed_{seed}' / 'train/last.pt'
            actor, saved = load_actor(checkpoint)
            for scenario in SCENARIOS:
                verify()
                seeds, horizon = evaluation_protocol(scenario)
                layout = layouts().get(scenario) if scenario not in ('reference', 'long_run') else None
                metadata = dict(checkpoint_metadata(checkpoint, saved, 'cpu', layout), method=mode, training_seed=seed,
                                checkpoint_kind='last', scenario=scenario)
                inputs = dict(metadata=metadata, seeds=seeds, steps=horizon)
                target = output / 'evaluation' / mode / f'seed_{seed}' / f'last_{scenario}.jsonl'
                if not cached(target, inputs):
                    rows = evaluate_actor(actor, saved['config']['env_name'], seeds, horizon, scenario == 'long_run',
                                          True, layout, metadata)
                    if sha(checkpoint) != metadata['checkpoint_sha256'] or not all(torch.equal(v, saved['actor'][k]) for k, v in actor.state_dict().items()):
                        raise ValueError('Pilot evaluation changed checkpoint parameters')
                    verify()
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.with_name(target.name + '.partial').open('x') as handle:
                        for row in rows:
                            handle.write(json.dumps(row, allow_nan=False) + '\n')
                    finish(target, inputs)
                rows = read_rows(target)
                if [r['seed'] for r in rows] != seeds or any(r['steps'] != horizon for r in rows):
                    raise ValueError(f'Incomplete pilot evaluation: {target}')
                log(f'Evaluated {mode}, seed {seed}, {scenario}')


def pipeline(prior, output, log, verify):
    from report_ccpd_investigation import analyze_existing, generate_report
    actions = (lambda: analyze_existing(prior, output), lambda: replay_failures(prior, output, log),
               lambda: audit_selections(prior, output, log), lambda: train_pilot(prior, output, log, verify),
               lambda: evaluate_pilot(output, log, verify), lambda: generate_report(prior, output))
    for stage, action in zip(STAGES, actions):
        verify()
        write_json(output / 'status.json', dict(stage=stage, updated_at=time.time(), pid=os.getpid()))
        log(f'Stage: {stage}')
        action()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, default=ORIGINAL / 'results/ccpd_diagnostic')
    parser.add_argument('--output', type=Path, default=ORIGINAL / 'results/ccpd_investigation')
    parser.add_argument('--check', action='store_true', help='Tests and 4096-step CPU smoke only; no full experiment')
    parser.add_argument('--dry-run', action='store_true', help='Show protocol without creating files or starting processes')
    args = parser.parse_args()
    prior, output = args.prior.resolve(), args.output.resolve()
    assert_isolation()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.dry_run:
        print(json.dumps(dict(isolated_checkout=str(ROOT), simulator=str(ROOT / 'robotic-warehouse/rware'),
                              output=str(output), prerequisite=str(prior), stages=STAGES,
                              modes=MODES, training_seeds=[0, 1], steps_per_run=BUDGET, total_steps=8*BUDGET,
                              episodes={s: evaluation_protocol(s) for s in SCENARIOS}), indent=2))
        return
    if output == prior or output.is_relative_to(prior):
        parser.error('Output must be separate from the existing experiment')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                            handlers=[logging.StreamHandler(), logging.FileHandler(output / 'pipeline.log')], force=True)
        log = logging.info
        try:
            if args.check:
                run_checks(output, log)
                return
            write_json(output / 'status.json', dict(stage='waiting_for_prerequisite', updated_at=time.time(), pid=os.getpid()))
            wait_prerequisite(prior, log)
            original_hashes = verify_prior(prior)
            run_checks(output, log)
            sources = source_hashes()
            manifest = dict(source_hashes=sources, prerequisite_hashes=original_hashes, prior=str(prior),
                            budget=BUDGET, modes=list(MODES), training_seeds=[0, 1],
                            packages={n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'gymnasium')})
            path = output / 'manifest.json'
            if path.exists() and json.loads(path.read_text()) != manifest:
                raise ValueError('Experiment inputs/code changed; use a fresh --output directory')
            if not path.exists():
                write_json(path, manifest)
            def verify():
                if source_hashes() != sources:
                    raise ValueError('Isolated experiment source changed during execution')
            pipeline(prior, output, log, verify)
            verify()
            write_json(output / 'status.json', dict(stage='complete', updated_at=time.time(), pid=os.getpid()))
            log(f'Complete: {output / "comparison.md"}; {output / "next_steps.md"}')
        except BaseException as error:
            write_json(output / 'status.json', dict(stage='failed', error=str(error), updated_at=time.time(), pid=os.getpid()))
            log(f'Failed: {error}. Completed artifacts are preserved.')
            raise


if __name__ == '__main__':
    main()
