import importlib.util
import json
from pathlib import Path
import sys
import torch
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import pretrained


def test_integrity_and_report():
    pretrained.report(pretrained.verify())

@pytest.mark.parametrize('method', ['local', 'communicating_ccpd'])
@pytest.mark.parametrize('agents', [40, 50])
@pytest.mark.parametrize('kind', ['best', 'last'])
def test_export_matches_training_checkpoint_and_runs(method, agents, kind):
    from run_stage3 import load_checkpoint, load_layout, NavigationEpisode, stream_seed
    from navigation_policy import PolicyRuntime
    torch.set_num_threads(1)
    p=ROOT/'pretrained'/method/f'n{agents}'
    full,_=load_checkpoint(p/f'{kind}.pt')
    actor=PolicyRuntime(p/f'actor_{kind}.pt').actor
    for name,value in actor.state_dict().items():
        torch.testing.assert_close(value,full.state_dict()[name],rtol=0,atol=0)
    ep=NavigationEpisode(load_layout(),200001,210001,8,(21,41),agents,actor.communication,.1,stream_seed(200001,0,0xC4))
    try:
        actor.seed_streams(stream_seed(200001,0,0xA4),agents)
        full.seed_streams(stream_seed(200001,0,0xA4),agents)
        state=actor.initial_state((agents,),'cpu');other=state.clone()
        with torch.no_grad():
            for _ in range(8):
                obs=torch.as_tensor(ep.obs)
                action,logp,state=actor.act(obs,state)
                expected,logp2,other=full.act(obs,other)
                torch.testing.assert_close(action,expected,rtol=0,atol=0)
                torch.testing.assert_close(logp,logp2,rtol=0,atol=0)
                assert torch.isfinite(state).all()
                ep.step(action.numpy())
        assert ep.t==8
    finally:ep.close()

def test_seed0_wrapper_preserves_scope_in_child_processes():
    import subprocess
    code = """
import runpy
from types import SimpleNamespace
from pathlib import Path
ns=runpy.run_path('scripts/train_seed0.py',run_name='release_test')
r=ns['runner']
assert r.SEEDS==(0,)
assert r.source_path('local',0).is_file()
a=SimpleNamespace(output=Path('/tmp/not-launched'),schedule=((1,2),),smoke=True)
c=r.child_command(a,'profile',condition='local',seed=0,agents=50,device='cpu')
assert c[2].endswith('/scripts/train_seed0.py')
"""
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)
    p=subprocess.run([sys.executable,str(ROOT/'scripts/train_seed0.py'),'--campaign'],capture_output=True,text=True)
    assert p.returncode!=0 and 'Holdout requires' in p.stderr
