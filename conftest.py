"""Point historical regression fixtures at released Stage-1 sources, without changing runtime code."""
from pathlib import Path
import os
import subprocess
import sys
import pytest
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'stage3_runtime/robotic-warehouse'), str(ROOT/'stage3_runtime/seac/seac'), str(ROOT/'stage3_runtime/scripts')]

@pytest.fixture(autouse=True)
def released_sources(request, monkeypatch):
    paths = [str(ROOT/'stage3_runtime/robotic-warehouse'), str(ROOT/'stage3_runtime/seac/seac'), str(ROOT/'stage3_runtime/scripts')]
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join(paths + [os.environ.get('PYTHONPATH', '')]))
    if request.node.name in {'test_original_evaluation_parity_and_recurrent_reset', 'test_historical_disabled_learner_parity'}:
        probe = subprocess.run(['git', 'cat-file', '-e', '2e8d522^{commit}'], cwd=ROOT, capture_output=True)
        if probe.returncode:
            pytest.skip('Historical baseline commit 2e8d522 is unavailable in this shallow/archive checkout')
    runner = getattr(request.module, 'runner', None)
    if runner is None:
        return
    if hasattr(runner, 'source_path'):
        monkeypatch.setattr(runner, 'source_path', lambda method, seed: ROOT / 'pretrained/stage1' / method / f'seed_{seed}/last.pt')
    if hasattr(runner, 'source_checkpoint'):
        monkeypatch.setattr(runner, 'source_checkpoint', lambda method: ROOT / 'pretrained/stage1' / method / 'seed_0/last.pt')
