"""Real stand-in subprocesses exercise launcher ownership and failure cleanup."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/run_shared_baselines.py'
spec = importlib.util.spec_from_file_location('shared_launcher', SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def job(tmp_path, name, code, gpu):
    directory = tmp_path / name
    directory.mkdir()
    return dict(run_dir=str(directory), command=[sys.executable, '-c', code], cwd=str(directory),
                environment={'CUDA_VISIBLE_DEVICES': gpu, 'PHYSICAL_GPU_ID': gpu})


def record(j):
    return json.loads((Path(j['run_dir']) / 'launch.json').read_text())


def test_concurrent_lifetime_and_isolated_visibility(tmp_path):
    code = 'import os,time,json; print(json.dumps(dict(start=time.time(), gpu=os.environ["CUDA_VISIBLE_DEVICES"])),flush=True); time.sleep(.4); print(time.time())'
    jobs = [job(tmp_path, str(i), code, str(i)) for i in range(2)]
    assert launcher.run_wave(jobs) == 0
    windows = []
    for i, j in enumerate(jobs):
        lines = (Path(j['run_dir']) / 'train.log').read_text().splitlines()
        start = json.loads(lines[0])
        assert start['gpu'] == str(i)
        windows.append((start['start'], float(lines[1])))
        assert record(j)['exit_status'] == 0
    assert max(w[0] for w in windows) < min(w[1] for w in windows)


def test_failure_stops_sibling(tmp_path):
    jobs = [job(tmp_path, 'failure', 'import time; time.sleep(.2); raise SystemExit(7)', '0'),
            job(tmp_path, 'sibling', 'import time; time.sleep(30)', '1')]
    assert launcher.run_wave(jobs) == 1
    assert record(jobs[0])['exit_status'] == 7
    assert record(jobs[1])['exit_status'] < 0


def test_spawn_failure_reaps_started_sibling(tmp_path):
    jobs = [job(tmp_path, 'sibling', 'import time; time.sleep(30)', '0'),
            job(tmp_path, 'failure', '', '1')]
    jobs[1]['command'] = ['/does-not-exist/shared-test-python']
    with pytest.raises(FileNotFoundError):
        launcher.run_wave(jobs)
    assert record(jobs[0])['exit_status'] < 0
    assert record(jobs[1])['exit_status'] == -1


def test_signal_cleanup_leaves_unrelated_process_alive(tmp_path):
    jobs = [job(tmp_path, 'child', 'import time; time.sleep(30)', '0')]
    jobs_file = tmp_path / 'jobs.json'
    jobs_file.write_text(json.dumps(jobs))
    code = ('import importlib.util,json,sys; '
            's=importlib.util.spec_from_file_location("launcher",sys.argv[1]); '
            'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
            'm.run_wave(json.load(open(sys.argv[2])))')
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    process = subprocess.Popen([sys.executable, '-c', code, str(SCRIPT), str(jobs_file)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 5
        while not (Path(jobs[0]['run_dir']) / 'launch.json').exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(.02)
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=8)
        assert process.returncode != 0
        assert record(jobs[0])['exit_status'] < 0
        assert unrelated.poll() is None
        with pytest.raises(ProcessLookupError):
            os.kill(record(jobs[0])['pid'], 0)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        unrelated.terminate()
        unrelated.wait()


def test_dry_run_waves_venv_and_no_side_effects(tmp_path, capsys):
    output = tmp_path / 'does-not-exist'
    assert launcher.main(['--dry-run', '--output', str(output)]) == 0
    text = capsys.readouterr().out
    assert text.count('wave=1') == 2 and text.count('wave=2') == 2
    assert '.venv/bin/python' in text and text.count('device=cuda:0') == 4
    assert not output.exists()
    assert launcher.main(['--dry-run', '--jobs-per-gpu', '2', '--output', str(output)]) == 0
    assert capsys.readouterr().out.count('wave=1') == 4
    assert launcher.main(['--dry-run', '--models', 'mappo_gru_ccpd_routing', '--seeds', '0', '1', '2',
                          '--num-env-steps', '20000000', '--output', str(output)]) == 0
    commands = capsys.readouterr().out
    assert commands.count('mappo_gru_ccpd_routing') == 6  # config and output path, per seed
    assert commands.count('physical_gpu=1') == 3 and not output.exists()
    assert not output.exists()


def test_existing_attempt_rejected_before_gpu_access(tmp_path):
    existing = tmp_path / 'shared_ppo_routing/seed_0/attempt_1'
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        launcher.main(['--output', str(tmp_path)])
    assert not (tmp_path / 'mappo_routing').exists()


def test_busy_and_visibility_preflight(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    def query(command, **kwargs):
        return '0, GPU-A\n1, GPU-B\n' if 'index,uuid' in command[1] else 'GPU-A, 12, 20 MiB\n'
    monkeypatch.setattr(launcher.subprocess, 'check_output', query)
    with pytest.raises(RuntimeError, match='busy'):
        launcher.resolve_gpus(['0', '1'], False)
    assert launcher.resolve_gpus(['0', '1'], True) == {'0': 'GPU-A', '1': 'GPU-B'}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='inherited'):
        launcher.resolve_gpus(['0', '1'], True)


def test_ccpd_three_seeds_share_one_concurrent_wave(tmp_path, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    import run_ccpd_seeds as ccpd
    import run_shared_baselines as common
    output = tmp_path / 'ccpd'
    def no_gpu(*args):
        raise AssertionError('Dry run must not probe GPUs')
    monkeypatch.setattr(common, 'resolve_gpus', no_gpu)
    assert ccpd.main(['--dry-run', '--output', str(output)]) == 0
    text = capsys.readouterr().out
    assert 'seed=0 physical_gpu=0' in text and 'seed=1 physical_gpu=1' in text
    assert 'seed=2 physical_gpu=1' in text and text.count('num_env_steps=20000000') == 3
    assert not output.exists()
    monkeypatch.setattr(common, 'resolve_gpus', lambda *args: {'0': 'GPU-A', '1': 'GPU-B'})
    probes = []
    monkeypatch.setattr(common.subprocess, 'run', lambda *args, **kwargs: probes.append(kwargs['env']['CUDA_VISIBLE_DEVICES']))
    def real_wave_with_standins(jobs):
        assert [(j['seed'], j['physical_gpu']) for j in jobs] == [(0, '0'), (1, '1'), (2, '1')]
        for j in jobs:
            assert 'ccpd_trace_events=True' in j['command'] and 'num_env_steps=32' in j['command']
            j['command'] = [sys.executable, '-c',
                'import time,os,json; print(json.dumps(dict(start=time.time(),gpu=os.environ["CUDA_VISIBLE_DEVICES"])),flush=True); time.sleep(.4); print(time.time())']
        assert common.run_wave(jobs) == 0
        windows = []
        for j, gpu in zip(jobs, ('GPU-A', 'GPU-B', 'GPU-B')):
            lines = (Path(j['run_dir']) / 'train.log').read_text().splitlines()
            start = json.loads(lines[0])
            assert start['gpu'] == gpu and record(j)['exit_status'] == 0
            windows.append((start['start'], float(lines[1])))
        assert max(start for start, _ in windows) < min(end for _, end in windows)
        return 0
    monkeypatch.setattr(ccpd, 'run_wave', real_wave_with_standins)
    assert ccpd.main(['--output', str(output), '--num-env-steps', '32']) == 0
    assert probes == ['GPU-A', 'GPU-B']
    with pytest.raises(FileExistsError):
        ccpd.main(['--output', str(output)])
