"""Point historical regression fixtures at released Stage-1 sources, without changing runtime code."""
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'stage3_runtime/robotic-warehouse'), str(ROOT/'stage3_runtime/seac/seac'), str(ROOT/'stage3_runtime/scripts')]

@pytest.fixture(autouse=True)
def released_sources(request, monkeypatch):
    runner = getattr(request.module, 'runner', None)
    if runner is None:
        return
    if hasattr(runner, 'source_path'):
        monkeypatch.setattr(runner, 'source_path', lambda method, seed: ROOT / 'pretrained/stage1' / method / f'seed_{seed}/last.pt')
    if hasattr(runner, 'source_checkpoint'):
        monkeypatch.setattr(runner, 'source_checkpoint', lambda method: ROOT / 'pretrained/stage1' / method / 'seed_0/last.pt')
