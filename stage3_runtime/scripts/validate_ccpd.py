#!/usr/bin/env python3
"""Frozen CCPD diagnostic protocol: audit, evaluation, then matched ablations.

Outputs are atomic and restartable. A completed evaluation is reused only when
its checkpoint, evaluator, environment and episode protocol match exactly.
"""
import argparse
import ast
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'seac/seac'))
from rware import CUSTOM_5_ROBOT_LAYOUT
from shared_ccpd import DEFAULTS as CCPD_DEFAULTS

SEEDS = list(range(2000, 2050))
SCENARIOS = ('reference', 'horizontal', 'vertical', 'rotation', 'long_run')
METHODS = ('ccpd', 'mappo', 'shared_ppo')


def source_hashes():
    paths = [*list((ROOT / 'seac/seac').glob('*.py')),
             *list((ROOT / 'seac/seac/configs').glob('*.yaml')),
             *list((ROOT / 'robotic-warehouse/rware').glob('*.py')),
             ROOT / 'scripts/validate_ccpd.py', ROOT / 'scripts/summarize_ccpd_validation.py',
             ROOT / 'scripts/run_shared_baselines.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def verify_frozen_sources(output):
    path = output / 'protocol_sources.json'
    if path.exists() and json.loads(path.read_text()) != source_hashes():
        raise ValueError('Protocol source changed during experiment; use a new output directory')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def training_dir(method, seed, output):
    if method in ('random', 'all_conflict'):
        return output / 'training' / method / f'seed_{seed}' / 'train'
    model = {'ccpd': 'mappo_gru_ccpd_routing', 'mappo': 'mappo_gru_routing',
             'shared_ppo': 'shared_ppo_gru_routing'}[method]
    family, attempt = ('ccpd', 'ccpd_v0_20m') if method == 'ccpd' else ('shared_baselines', 'four_models_20m')
    return ROOT / 'results' / family / model / f'seed_{seed}' / attempt / 'train'


def layouts():
    rows = CUSTOM_5_ROBOT_LAYOUT.strip().splitlines()
    variants = dict(reference=rows, horizontal=[r[::-1] for r in rows],
                    vertical=rows[::-1], rotation=[r[::-1] for r in rows[::-1]])
    return {name: '\n'.join(value) + '\n' for name, value in variants.items()}


def validate_layout(layout):
    rows = layout.strip().splitlines()
    if len(rows) != 10 or any(len(row) != 6 for row in rows):
        raise ValueError('Expected a 10x6 layout')
    if set(''.join(rows)) - set('.xg'):
        raise ValueError('Unsupported map symbols')
    racks = {(x, y) for y, row in enumerate(rows) for x, c in enumerate(row) if c == 'x'}
    goals = {(x, y) for y, row in enumerate(rows) for x, c in enumerate(row) if c == 'g'}
    if len(racks) != 21 or len(goals) != 1:
        raise ValueError('Expected 21 racks and one workstation')
    # Loaded robots cannot cross other racks. Every rack needs a service route.
    for rack in racks:
        queue, seen = deque([rack]), {rack}
        while queue:
            x, y = queue.popleft()
            for p in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if 0 <= p[0] < 6 and 0 <= p[1] < 10 and p not in racks and p not in seen:
                    seen.add(p)
                    queue.append(p)
        if not goals <= seen:
            raise ValueError(f'Rack {rack} has no loaded service route')
    return dict(height=10, width=6, racks=len(racks), workstations=len(goals), all_routes_valid=True)


def git_source(revision, path):
    return subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=ROOT)


def normalized_simulator(source):
    """Remove only the audited, opt-in trace instrumentation for AST parity."""
    tree = ast.parse(source)
    class StripTrace(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.unparse(node.test).startswith('self.coordination_trace_enabled'):
                return None
            return self.generic_visit(node)
        def visit_Assign(self, node):
            if any(ast.unparse(t) == 'self.coordination_trace_enabled' for t in node.targets):
                return None
            return self.generic_visit(node)
        def visit_FunctionDef(self, node):
            if node.name == '__init__' and node.args.args[-1].arg == 'coordination_trace_enabled':
                node.args.args.pop()
                node.args.defaults.pop()
            return self.generic_visit(node)
    return ast.dump(StripTrace().visit(tree), include_attributes=False)


def audit(output):
    import torch
    from evaluate_shared import checkpoint_metadata
    import rware
    if Path(rware.__file__).resolve() != ROOT / 'robotic-warehouse/rware/__init__.py':
        raise ValueError('Python is importing a different simulator installation')
    reference = json.loads((training_dir('ccpd', 0, output) / 'provenance.json').read_text())
    current_sim = (ROOT / 'robotic-warehouse/rware/warehouse.py').read_bytes()
    if normalized_simulator(current_sim) != normalized_simulator(git_source('2e8d522', 'robotic-warehouse/rware/warehouse.py')):
        raise ValueError('Simulator semantics differ beyond audited trace instrumentation')
    items = []
    ignored = {'method', 'seed', 'device', 'run_dir', 'ccpd_mode', 'ccpd_trace_events'}
    for method in METHODS:
        for seed in range(3):
            directory = training_dir(method, seed, output)
            p = json.loads((directory / 'provenance.json').read_text())
            config = dict(CCPD_DEFAULTS, **p['config'])
            baseline = dict(CCPD_DEFAULTS, **reference['config'])
            mismatch = [k for k in baseline if k not in ignored and config.get(k) != baseline[k]]
            if mismatch or config['seed'] != seed:
                raise ValueError(f'{directory}: incompatible settings {mismatch}')
            expected_method = 'shared_ppo' if method == 'shared_ppo' else 'mappo'
            expected_mode = 'successful' if method == 'ccpd' else 'off'
            if config['method'] != expected_method or config['ccpd_mode'] != expected_mode:
                raise ValueError(f'{directory}: wrong method/mode')
            for k in ('architecture', 'state_schema', 'map_sha256', 'actual_layout', 'packages'):
                if p[k] != reference[k]:
                    raise ValueError(f'{directory}: mismatched {k}')
            if p['architecture'] != dict(obs_size=199, actions=4, state_size=320, recurrent=True):
                raise ValueError('Unexpected architecture')
            snapshot = directory / 'source/warehouse.py'
            simulator = snapshot.read_bytes() if snapshot.exists() else git_source(p['git_commit'], 'robotic-warehouse/rware/warehouse.py')
            if not snapshot.exists() and ('robotic-warehouse/' in p['git_status'] or (directory / 'source.diff').read_text().strip()):
                raise ValueError('Cannot reconstruct dirty historical simulator')
            if normalized_simulator(simulator) != normalized_simulator(current_sim):
                raise ValueError(f'{directory}: historical simulator behavior differs')
            # Registration supplies layout and overrides; it must also be unchanged.
            if git_source(p['git_commit'], 'robotic-warehouse/rware/__init__.py') != (ROOT / 'robotic-warehouse/rware/__init__.py').read_bytes():
                raise ValueError('Historical environment registration differs')
            for name in ('shared_models.py', 'shared_storage.py'):
                if (directory / 'source' / name).read_bytes() != (ROOT / 'seac/seac' / name).read_bytes():
                    raise ValueError(f'{directory}: changed {name}')
            for name in ('shared_ppo.py', 'shared_ccpd.py'):
                archived = directory / 'source' / name
                if method == 'ccpd' and archived.read_bytes() != (ROOT / 'seac/seac' / name).read_bytes():
                    raise ValueError(f'{directory}: CCPD algorithm changed')
                if method != 'ccpd' and name == 'shared_ppo.py' and archived.read_bytes() != git_source(p['git_commit'], 'seac/seac/shared_ppo.py'):
                    raise ValueError('Unrecognized baseline learner')
            checkpoints = {}
            final_saved = None
            for kind in ('last', 'best'):
                path = directory / f'{kind}.pt'
                saved = torch.load(path, map_location='cpu', weights_only=False)
                if saved['config'] != p['config'] or saved['architecture'] != p['architecture']:
                    raise ValueError(f'{path}: checkpoint/provenance mismatch')
                if kind == 'last' and not 20_000_000 <= saved['env_steps'] < 20_002_048:
                    raise ValueError(f'{path}: incorrect completed budget')
                if kind == 'last':
                    final_saved = saved
                checkpoints[kind] = dict(sha256=sha(path), env_steps=saved['env_steps'])
            rows = [json.loads(line) for line in (directory / 'evaluation.jsonl').read_text().splitlines()]
            final_step = max(r['env_steps'] for r in rows)
            final = [r for r in rows if r['env_steps'] == final_step]
            if sorted(r['seed'] for r in final) != [1000, 1001, 1002, 1003]:
                raise ValueError('Unexpected final validation suite')
            items.append(dict(method=method, training_seed=seed, directory=str(directory),
                              checkpoints=checkpoints, config=config, simulator_sha256=hashlib.sha256(simulator).hexdigest(),
                              simulator_source='snapshot' if snapshot.exists() else 'clean recorded git revision',
                              original_report_source='final training validation, seeds 1000–1003',
                              final_validation={k: sum(r[k] for r in final) / len(final) for k in
                                                ('cycles_per_1000_steps', 'conflict_attempts', 'movement_denied')},
                              evaluation_metadata=checkpoint_metadata(directory / 'last.pt', final_saved, 'cpu')))
    report = dict(status='passed', runs=items,
                  comparability='Same configuration except method/seed/operational fields and CCPD mode. Simulator AST equals historical baseline after removing only opt-in trace instrumentation. Disabled-CCPD learner parity covered by regression tests.',
                  layouts={k: validate_layout(v) for k, v in layouts().items()},
                  reserved_seeds='3000–3049: not evaluated',
                  limitations=['Three training seeds give limited precision.', 'CPU inference timings under concurrent evaluation are diagnostic, not an isolated latency benchmark.'])
    write_json(output / 'audit.json', report)
    for name, layout in layouts().items():
        path = output / 'layouts' / f'{name}.map'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != layout:
            raise ValueError('Existing layout changed')
        path.write_text(layout)
    return report


def evaluation_job(job):
    import torch
    from evaluate_shared import load_actor, evaluate_actor, checkpoint_metadata
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    method, seed, kind, scenario, directory, output = job
    verify_frozen_sources(output)
    checkpoint = directory / f'{kind}.pt'
    layout = layouts().get(scenario) if scenario not in ('reference', 'long_run') else None
    horizon = 10000 if scenario == 'long_run' else 500
    target = output / 'evaluation' / method / f'seed_{seed}' / f'{kind}_{scenario}.jsonl'
    actor, saved = load_actor(checkpoint)
    fingerprint = hashlib.sha256(b''.join((ROOT / p).read_bytes() for p in
        ('seac/seac/evaluate_shared.py', 'seac/seac/shared_envs.py', 'scripts/validate_ccpd.py'))).hexdigest()
    metadata = dict(checkpoint_metadata(checkpoint, saved, 'cpu', layout), method=method,
                    checkpoint_kind=kind, scenario=scenario, evaluator_sha256=fingerprint)
    if target.exists():
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        if (len(rows) != 50 or [r['seed'] for r in rows] != SEEDS
                or any(any(r.get(k) != v for k, v in metadata.items()) or r['requested_steps'] != horizon
                       or not r['deterministic'] or r['continuous'] != (scenario == 'long_run') for r in rows)):
            raise ValueError(f'Existing evaluation has incompatible provenance: {target}')
        return str(target)
    rows = evaluate_actor(actor, saved['config']['env_name'], SEEDS, horizon,
                          scenario == 'long_run', True, layout, metadata)
    if sha(checkpoint) != metadata['checkpoint_sha256']:
        raise ValueError('Checkpoint changed during evaluation')
    verify_frozen_sources(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
    temporary.replace(target)
    return str(target)


def evaluate_suite(output, methods, reference_only=False, workers=3):
    jobs = []
    for method in methods:
        for seed in range(3):
            directory = training_dir(method, seed, output)
            for kind in ('last', 'best'):
                scenarios = ('reference',) if kind == 'best' or reference_only else SCENARIOS[1:]
                for scenario in scenarios:
                    jobs.append((method, seed, kind, scenario, directory, output))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(evaluation_job, jobs):
            print(f'Evaluated {result}', flush=True)


def train_controls(output):
    from run_shared_baselines import prepare_jobs, run_wave
    python = ROOT / '.venv/bin/python'
    template = json.loads((training_dir('ccpd', 0, output) / 'provenance.json').read_text())['config']
    for seed in range(3):
        verify_frozen_sources(output)
        jobs = []
        for mode, gpu in (('random', '0'), ('all_conflict', '1')):
            directory = training_dir(mode, seed, output)
            config = dict(template, seed=seed, ccpd_mode=mode, run_dir=str(directory), device='cuda:0')
            if directory.exists():
                summary = directory / 'summary.json'
                provenance = directory / 'provenance.json'
                if (not summary.exists() or json.loads(summary.read_text())['env_steps'] < 20_000_000
                        or not provenance.exists() or json.loads(provenance.read_text())['config'] != config):
                    raise ValueError(f'Incomplete/incompatible training requires explicit recovery: {directory}')
                continue
            config_path = output / 'configs' / f'{mode}_{seed}.json'
            write_json(config_path, config)
            jobs.append(dict(model=mode, seed=seed, physical_gpu=gpu, run_dir=str(directory.parent),
                             command=[str(python), '-u', str(ROOT / 'seac/seac/train_shared.py'), 'with', str(config_path)],
                             cwd=str(ROOT / 'seac/seac')))
        if jobs:
            prepare_jobs(jobs, ['0', '1'], False, python)
            if run_wave(jobs):
                raise RuntimeError('Ablation training failed; inspect launch.json and train.log')


def audit_controls(output):
    import torch
    template = json.loads((training_dir('ccpd', 0, output) / 'provenance.json').read_text())['config']
    records = []
    for mode in ('random', 'all_conflict'):
        for seed in range(3):
            directory = training_dir(mode, seed, output)
            expected = dict(template, seed=seed, ccpd_mode=mode, run_dir=str(directory), device='cuda:0')
            provenance = json.loads((directory / 'provenance.json').read_text())
            if provenance['config'] != expected or provenance['resume_restarts_environments']:
                raise ValueError(f'Control did not train from matched fresh configuration: {directory}')
            for filename in ('shared_ppo.py', 'shared_ccpd.py', 'shared_models.py', 'shared_storage.py', 'warehouse.py'):
                original = ROOT / ('robotic-warehouse/rware' if filename == 'warehouse.py' else 'seac/seac') / filename
                if sha(directory / 'source' / filename) != sha(original):
                    raise ValueError(f'Control algorithm/simulator changed: {filename}')
            hashes = {}
            for kind in ('last', 'best'):
                checkpoint = directory / f'{kind}.pt'
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if saved['config'] != expected or saved['architecture'] != provenance['architecture']:
                    raise ValueError('Control checkpoint/provenance mismatch')
                if kind == 'last' and not 20_000_000 <= saved['env_steps'] < 20_002_048:
                    raise ValueError('Control training budget incomplete')
                hashes[kind] = sha(checkpoint)
            records.append(dict(mode=mode, training_seed=seed, checkpoints=hashes))
    write_json(output / 'ablation_audit.json', dict(status='passed', runs=records))


def status(output, stage, **extra):
    write_json(output / 'status.json', dict(stage=stage, updated_at=time.time(), **extra))


def validate_pipeline(output):
    import os
    environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                       PYTHONPATH=str(ROOT / 'robotic-warehouse'))
    python = ROOT / '.venv/bin/python'
    hashes = source_hashes()
    hashes.update({str(p.relative_to(ROOT)): sha(p) for folder in ('seac/tests', 'robotic-warehouse/tests')
                   for p in (ROOT / folder).glob('*.py')})
    validation = output / 'validation.json'
    cached = json.loads(validation.read_text()) if validation.exists() else {}
    if cached.get('source_hashes') != hashes or cached.get('status') != 'passed':
        with (output / 'tests.log').open('w') as log:
            subprocess.run([str(python), '-m', 'pytest', 'seac/tests', 'robotic-warehouse/tests', '-q'],
                           cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    for mode in ('random', 'all_conflict'):
        directory = output / 'smoke' / mode
        if not directory.exists():
            directory.parent.mkdir(parents=True, exist_ok=True)
            with (directory.parent / f'{mode}.log').open('w') as log:
                subprocess.run([str(python), str(ROOT / 'seac/seac/train_shared.py'), 'with',
                                'mappo_gru_ccpd_routing', f'ccpd_mode={mode}', 'device=cpu',
                                'num_env_steps=4096', 'ccpd_trace_events=True', f'run_dir={directory}'],
                               cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(s) for s in (directory / 'metrics.jsonl').read_text().splitlines()]
        rows = [r for r in rows if r.get('kind') == 'learning']
        import math
        if (not rows or not (directory / 'summary.json').exists()
                or not all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v, (int, float)))
                or sum(r.get('ccpd_selected_samples', 0) for r in rows) <= 0
                or any(r['ccpd_selected_noop_fraction'] > .25 + 1e-8
                       or r['ccpd_selected_fraction'] > .1 + 1e-8 for r in rows)):
            raise ValueError(f'Invalid smoke diagnostics: {mode}')
    write_json(output / 'validation.json', dict(status='passed', smoke_steps_per_mode=4096,
                                               tests_log=str(output / 'tests.log'), source_hashes=hashes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('audit', 'check', 'reference', 'transfer', 'train', 'controls', 'all'))
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ccpd_diagnostic')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.workers < 1:
        parser.error('workers must be positive')
    lock = output / 'pipeline.lock'
    import fcntl
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status(output, 'audit')
            audit(output)
            if args.stage != 'audit':
                status(output, 'pipeline_validation')
                validate_pipeline(output)
            if args.stage not in ('audit', 'check'):
                verify_frozen_sources(output)
                write_json(output / 'protocol_sources.json', source_hashes())
            from summarize_ccpd_validation import summarize
            if args.stage in ('reference', 'all'):
                status(output, 'held_out_evaluation')
                evaluate_suite(output, METHODS, True, args.workers)
                summarize(output)
            if args.stage in ('transfer', 'all'):
                status(output, 'transfer_evaluation')
                evaluate_suite(output, METHODS, False, args.workers)
                summarize(output)
            if args.stage in ('train', 'all'):
                current = summarize(output)
                if any(s.split('/')[-1] in METHODS for s in current['missing']):
                    raise ValueError('Complete held-out and transfer evaluation before ablation training')
                status(output, 'ablation_training')
                train_controls(output)
            if args.stage in ('controls', 'all'):
                status(output, 'ablation_evaluation')
                audit_controls(output)
                evaluate_suite(output, ('random', 'all_conflict'), True, args.workers)
                evaluate_suite(output, ('random', 'all_conflict'), False, args.workers)
                result = summarize(output)
                if args.stage == 'all' and not result['complete']:
                    raise ValueError('The final report is missing required evaluations')
            status(output, 'complete' if args.stage == 'all' else f'{args.stage}_complete')
        except BaseException as error:
            status(output, 'failed', error=str(error))
            raise


if __name__ == '__main__':
    main()
