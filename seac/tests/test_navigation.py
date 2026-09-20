"""Navigation contracts: isolation, geometry, communication, PPO and deployment."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from navigation_envs import (NavigationEnvs, NavigationEpisode, MessageChannel, load_layouts, geometry_hash,
                             validate_layout, local_observations, privileged_state, stream_seed)
from navigation_policy import (NavigationActor, NavigationCritic, PolicyRuntime, export_actor,
                               packet_features, LOCAL_SIZE, PACKET_SIZE, SPEC)
from navigation_train import configuration, load_checkpoint, evaluate, verify_training, NavigationPPO

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'assets/navigation_layouts.json'
torch.set_num_threads(1)


def layouts():
    return load_layouts(MANIFEST)


def fixture_layout():
    text = '.xx....\n.......\n.xx....\n.......\n.x.....\n...g...\n'
    return dict(id='engineering-only', family='engineering-only', layout=text, shape=validate_layout(text),
                sha256=geometry_hash(text), weight=1.)


def episode(seed=2000, **kw):
    layout = layouts()[0]
    return NavigationEpisode(layout, seed, seed+10000, 100, layout['shape'], **kw)


def test_manifest_hash_routes_splits_and_reserved(tmp_path):
    data = json.loads(MANIFEST.read_text())
    data['layouts'][0]['path'] = str(ROOT/'assets/navigation_original.map')
    f = tmp_path/'layouts.json'; f.write_text(json.dumps(data))
    assert load_layouts(f)[0]['shape'] == (10, 6)
    with pytest.raises(ValueError, match='overlaps'):
        load_layouts(f, mode='cross_layout', training_families=['original'])
    data['layouts'][0]['sha256'] = 'bad'; f.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='hash'):
        load_layouts(f)
    for text in ('....\n..', '..g\n...', 'xxx\nxxx\nxxx\n..g'):
        with pytest.raises(ValueError):
            validate_layout(text)
    with pytest.raises(ValueError, match='Reserved'):
        episode(3000)
    assert stream_seed(0, 0, 0) > 3049


def fake_world(rotated=False, shift=(0, 0)):
    h, width = (7, 9)
    positions = [(3, 3), (4, 3), (3, 2)]
    goals = [(7, 5), (1, 1), (2, 5)]
    racks = [(2, 3), (5, 4)]
    headings = [0, 2, 1]
    if rotated:
        turn = lambda p: (h-1-p[1], p[0])
        positions, goals, racks = [list(map(turn, p)) for p in (positions, goals, racks)]
        headings = [{0:3, 3:1, 1:2, 2:0}[v] for v in headings]
        h, width = width, h
    positions = [(x+shift[0], y+shift[1]) for x,y in positions]
    goals = [(x+shift[0], y+shift[1]) for x,y in goals]
    racks = [(x+shift[0], y+shift[1]) for x,y in racks]
    h += shift[1]; width += shift[0]
    grid = np.zeros((2, h, width), int)
    for i, (x,y) in enumerate(positions): grid[0,y,x] = i+1
    for i, (x,y) in enumerate(racks): grid[1,y,x] = i+1
    agents = [NS(x=x,y=y,dir=NS(value=d),carrying_shelf=None) for (x,y),d in zip(positions,headings)]
    tasks = [NS(phase=NS(value=0), goal=g) for g in goals]
    return NS(grid_size=(h,width), grid=grid, agents=agents, task_manager=NS(tasks=tasks,target=lambda t:t.goal),
              _previous_actions=np.eye(4)[[1,2,3]], _movement_success=np.array([1,0,0]), _cycle_steps=np.array([8,9,10]))


def test_observation_rotation_translation_and_no_identity_leak():
    original = local_observations(fake_world(), [1,2,3], [0,1,0])
    rotated = local_observations(fake_world(True), [1,2,3], [0,1,0])
    shifted = local_observations(fake_world(shift=(2,2)), [1,2,3], [0,1,0])
    np.testing.assert_allclose(original, rotated, atol=1e-7)
    np.testing.assert_allclose(original, shifted, atol=1e-7)
    assert original.shape == (3, LOCAL_SIZE)
    w = fake_world(); w.agents[0].id=999; w.layout_id='secret'; w.seed=42
    np.testing.assert_array_equal(original, local_observations(w,[1,2,3],[0,1,0]))


def test_boundaries_goal_zero_and_stationarity():
    e = episode()
    try:
        e.w.agents[0].x=e.w.agents[0].y=0; e.w._recalc_grid(); e.refresh()
        assert (e.obs[0,:25] == 0).any()
        # Turning is not translation and does not clear stationary duration.
        for _ in range(40): e.step(np.full(5, 2))
        assert max(e.longest_stationary) >= 32
        assert e.metrics()['max_fleet_completion_gap'] >= 32
    finally: e.close()


def status(position, goal=(4,4)):
    return dict(position=list(position), heading=0, goal=list(goal), phase=0,
                carrying=False, previous_action=2, cycle_age=10)


def test_channel_delay_range_expiry_loss_and_isolation():
    statuses=[status((1,1)), status((2,1)), status((8,8))]
    channel=MessageChannel(3,12,loss=0)
    channel.transmit(statuses,0)
    args=([s['position'] for s in statuses],[0,0,0])
    assert channel.inboxes(*args,0).sum()==0
    assert channel.inboxes(*args,1)[... ,0].sum()==2
    assert channel.inboxes(*args,2)[... ,0].sum()==2
    assert channel.inboxes(*args,3).sum()==0
    other=MessageChannel(3,12,loss=0)
    assert other.inboxes(*args,1).sum()==0
    lost=MessageChannel(3,12,loss=1);lost.transmit(statuses,0)
    assert lost.inboxes(*args,1).sum()==0 and lost.dropped==2
    delayed=MessageChannel(3,12,loss=0,delay=2);delayed.transmit(statuses,0)
    assert delayed.inboxes(*args,1).sum()==0
    assert delayed.inboxes(*args,2)[...,0].sum()==2


def test_actor_neighbor_and_robot_permutations_empty_inbox():
    actor=NavigationActor(True)
    x=torch.randn(5, LOCAL_SIZE+4*(PACKET_SIZE+1)); x[:,LOCAL_SIZE::PACKET_SIZE+1]=1
    state=torch.randn(5,128)
    a,_=actor(x,state)
    y=x.clone();y[:,LOCAL_SIZE:]=x[:,LOCAL_SIZE:].reshape(5,4,-1)[:,[2,0,3,1]].reshape(5,-1)
    b,_=actor(y,state);torch.testing.assert_close(a,b)
    p=torch.tensor([2,0,4,1,3]);c,_=actor(x[p],state[p]);torch.testing.assert_close(a[p],c)
    x[:,LOCAL_SIZE::PACKET_SIZE+1]=0
    empty,_=actor(x,state);none,_=actor(x[:,:LOCAL_SIZE],state)
    torch.testing.assert_close(empty,none)
    x[:,LOCAL_SIZE+1:]=float('nan')
    # Runtime rejects nonfinite incoming payloads rather than accepting invalid packets.
    from navigation_policy import pack_observation
    with pytest.raises(ValueError): pack_observation(np.zeros(LOCAL_SIZE), [[float('nan')]*PACKET_SIZE])


def test_critic_padding_and_robot_order():
    e=episode()
    try:
        local=e.obs[:,:LOCAL_SIZE]; small=(10,6);large=(13,11)
        a=NavigationCritic(small);b=NavigationCritic(large);b.load_state_dict(a.state_dict())
        xs=torch.tensor(privileged_state(e.w,local,small))[None]
        xl=torch.tensor(privileged_state(e.w,local,large))[None]
        torch.testing.assert_close(a(xs)[0],b(xl)[0],atol=1e-6,rtol=1e-6)
        dirty=xl.clone();maps=dirty[:,:4*13*11].reshape(1,4,13,11)
        maps[:,:3]=torch.where(maps[:,3:4]>0,maps[:,:3],torch.randn_like(maps[:,:3]))
        torch.testing.assert_close(b(xl)[0],b(dirty)[0])
        offset=4*10*6;permuted=xs.clone();permuted[:,offset:]=xs[:,offset:].reshape(1,5,-1)[:,[2,0,4,1,3]].reshape(1,-1)
        torch.testing.assert_close(a(xs)[0],a(permuted)[0])
    finally:e.close()


@pytest.mark.parametrize('n',[3,7])
def test_variable_robot_counts_and_mixed_layouts(n):
    original=layouts()[0];small=fixture_layout()
    if n==7:
        small=dict(small,layout='.xx....\n.......\n.xx....\n.......\n.xxx...\n...g...\n')
        small.update(shape=validate_layout(small['layout'],n),sha256=geometry_hash(small['layout']))
    envs=NavigationEnvs([original,small],0,[1,2],True,n_agents=n)
    try:
        actor=NavigationActor(True);actor.seed_streams(0,2*n)
        before=envs.obs.copy();transition=envs.step(np.zeros((2,n),int))
        assert transition.truncated.tolist()==[True,False]
        assert not envs.envs[0].channel.pending and envs.envs[0].t==0
        assert envs.counters.tolist()==[2,1]
        assert transition.final_state.shape==envs.state.shape
        assert not np.array_equal(transition.final_obs[0],envs.obs[0])
        assert actor(torch.tensor(envs.obs))[0].shape==(2,n,4)
        assert NavigationCritic(envs.map_shape)(torch.tensor(envs.state))[0].shape==(2,1)
        seen=set()
        for _ in range(16):
            seen.update(e.layout['id'] for e in envs.envs);envs.step(np.zeros((2,n),int))
        assert len(seen)==2
    finally:envs.close()


def test_rng_task_spawn_and_evaluation_restoration():
    a=episode(2000);b=NavigationEpisode(layouts()[0],2000,12050,100,(10,6));c=episode(2100)
    try:
        assert a.initial['positions']==b.initial['positions'] and a.initial['headings']==b.initial['headings']
        assert a.initial['racks']!=b.initial['racks']
        d=NavigationEpisode(layouts()[0],2100,12000,100,(10,6))
        try:
            assert a.initial['racks']==d.initial['racks'] and a.initial['positions']!=d.initial['positions']
        finally:d.close()
    finally:a.close();b.close();c.close()
    actor=NavigationActor(True);actor.seed_streams(55,5)
    original_rng=[copy.deepcopy(r.bit_generator.state) for r in actor.rngs]
    np_state=np.random.get_state();torch_state=torch.get_rng_state().clone()
    first=evaluate(actor,layouts()[0],[2000],15);second=evaluate(actor,layouts()[0],[2000],15)
    for row in (first[0],second[0]): row.pop('inference_ms_per_fleet_step')
    assert first==second
    assert original_rng==[r.bit_generator.state for r in actor.rngs]
    np.testing.assert_array_equal(np_state[1],np.random.get_state()[1]);assert torch.equal(torch_state,torch.get_rng_state())


def test_runtime_batched_policy_parity_and_schema(tmp_path):
    env=episode(communication=True,loss=0);actor=NavigationActor(True)
    try:
        env.step(np.zeros(5,int));x=torch.tensor(env.obs)
        state=torch.randn(5,128)
        path=tmp_path/'actor.pt';export_actor(actor,path);runtime=PolicyRuntime(path)
        with torch.no_grad():logits,next_state=actor(x,state)
        for i in range(5):
            slots=env.obs[i,LOCAL_SIZE:].reshape(-1,PACKET_SIZE+1)
            inbox=slots[slots[:,0]>0,1:]
            from navigation_policy import pack_observation
            with torch.no_grad(): single,_=runtime.actor(torch.tensor(pack_observation(env.obs[i,:LOCAL_SIZE],inbox))[None],state[i:i+1])
            torch.testing.assert_close(logits[i],single[0],atol=1e-6,rtol=1e-6)
            action,hidden=runtime.act(env.obs[i,:LOCAL_SIZE],inbox,state[i].numpy(),np.random.default_rng(i))
            np.testing.assert_allclose(hidden,next_state[i].numpy(),atol=1e-6)
            assert 0<=action<4
        broken=torch.load(path,weights_only=False);broken['observation_spec']={};torch.save(broken,path)
        with pytest.raises(ValueError):PolicyRuntime(path)
        with pytest.raises(ValueError):load_checkpoint(path)
    finally:env.close()


@pytest.mark.parametrize('communication,ccpd',[(False,False),(True,False),(False,True),(True,True)])
def test_ppo_reconstruction_recorded_packets_finite_updates(communication,ccpd):
    env=NavigationEnvs(layouts(),0,[13,19],communication)
    config=configuration(MANIFEST,'communicating_ccpd' if communication and ccpd else 'local',0)
    config.update(communication=communication,ccpd_mode='successful' if ccpd else 'off',rollout_steps=32,num_envs=2,
                  ppo_epochs=1,num_minibatches=2,sequence_length=16,burn_in=8)
    learner=NavigationPPO(env.obs.shape[-1],4,env.state.shape[-1],config,torch.device('cpu'),
                      actor=NavigationActor(communication),critic=NavigationCritic(env.map_shape))
    learner.actor.seed_streams(0,10)
    try:
        data,_,_=learner.collect(env)
        assert learner.policy_diagnostics(data)['max_log_prob_error']<2e-6
        recorded=data['obs'].clone()
        for e in env.envs:e.channel.loss=1
        assert learner.policy_diagnostics(data)['max_log_prob_error']<2e-6
        metrics=learner.update(data)
        assert all(np.isfinite(v) for v in metrics.values())
        assert torch.equal(recorded,data['obs'])
        data,_,_=learner.collect(env)
        assert data['resets'].any()
        assert learner.policy_diagnostics(data)['max_log_prob_error']<2e-6
    finally:env.close()


def test_legacy_archived_replay_unchanged():
    source=Path('/home/utar/seac-rware/results/ccpd_diagnostic/evaluation/ccpd/seed_0/last_reference.jsonl')
    if not source.exists():pytest.skip('Archived experiment unavailable')
    from evaluate_shared import load_actor,evaluate_actor
    archived=json.loads(source.read_text().splitlines()[0])
    actor,saved=load_actor(archived['checkpoint'])
    actual,=evaluate_actor(actor,saved['config']['env_name'],[archived['seed']],500)
    for key,value in actual.items():
        if key=='inference_ms_per_fleet_step':continue
        if value is None:assert archived[key] is None
        else:np.testing.assert_allclose(value,archived[key],atol=1e-8,rtol=1e-8)


def test_launcher_defaults_receipts_and_gpu_visibility(tmp_path,monkeypatch,capsys):
    spec=importlib.util.spec_from_file_location('navigation_launcher',ROOT/'scripts/run_decentralized_navigation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.main(['--dry-run','--output',str(tmp_path/'new')])==0
    value=json.loads(capsys.readouterr().out);assert value['training_runs']==0 and not (tmp_path/'new').exists()
    assert module.main(['--train','--dry-run'])==0
    assert json.loads(capsys.readouterr().out)['budget']==240_000_000
    path=tmp_path/'artifact';path.write_text('partial')
    with pytest.raises(ValueError):module.receipt_valid(path,{})
    with pytest.raises(ValueError):verify_training(tmp_path,{})
    import run_shared_baselines as scheduler
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    def query(command,**kw):
        return '0, GPU-A\n1, GPU-B\n' if 'index,uuid' in command[1] else 'GPU-A, 12\n'
    monkeypatch.setattr(scheduler.subprocess,'check_output',query)
    assert scheduler.free_gpus()==[('1','GPU-B')]


def test_aggregation_no_cycles_and_paired_rng_blocks():
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from report_navigation import summary,differences,paired_arrays
    env=episode()
    try:
        env.step(np.zeros(5,int));base=env.metrics();base['inference_ms_per_fleet_step']=0
        rows=[dict(base,training_seed=s,episode_id=e,action_replicate=r) for s in range(3) for e in range(2) for r in range(3)]
        assert summary(rows)['mean_cycle_time'] is None
        a,b,_,_=paired_arrays(rows,rows);assert a.shape==(3,2,9)
        result=differences(rows,rows,draws=100)
        assert all(v['difference']==v['lower']==v['upper']==0 for v in result.values())
        with pytest.raises(ValueError):paired_arrays(rows,rows[:-1])
    finally:env.close()


def test_goal_units_reset_and_truncation_bootstrap():
    from navigation_policy import goal_features
    np.testing.assert_allclose(goal_features(np.array([3., 4.])), [.6, .8, 5/6])
    np.testing.assert_array_equal(goal_features(np.array([0., 0.])), [0, 0, 0])
    actor=NavigationActor()
    x=torch.randn(2,LOCAL_SIZE)
    a,h=actor(x,torch.randn(2,128),torch.ones(2,dtype=torch.bool))
    b,k=actor(x,torch.zeros(2,128))
    torch.testing.assert_close(a,b);torch.testing.assert_close(h,k)
    from shared_storage import gae
    rewards=torch.tensor([[1.,2.],[3.,4.]])
    values=torch.zeros_like(rewards)
    terminal=torch.tensor([[False,False],[False,True]])
    truncated=torch.tensor([[True,False],[False,False]])
    advantages,_=gae(rewards,values,torch.full_like(values,10),terminal,truncated,gamma=.5,lam=1)
    torch.testing.assert_close(advantages,torch.tensor([[6.,9.],[8.,4.]]))


def test_standalone_export_needs_no_simulator(tmp_path):
    import os,subprocess,sys
    path=tmp_path/'actor.pt';export_actor(NavigationActor(True),path)
    (tmp_path/'navigation_policy.py').write_bytes((ROOT/'seac/seac/navigation_policy.py').read_bytes())
    code="""
import sys
class BlockSimulator:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('rware','gymnasium','shared_ppo','navigation_envs'):
            raise RuntimeError('Simulator dependency: '+fullname)
sys.meta_path.insert(0,BlockSimulator())
from navigation_policy import PolicyRuntime,LOCAL_SIZE
import numpy as np
runtime=PolicyRuntime('actor.pt')
a,h=runtime.act(np.zeros(LOCAL_SIZE),[],None,np.random.default_rng(1))
assert 0<=a<4 and h.shape==(128,)
"""
    subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=dict(os.environ,PYTHONPATH=''),check=True)


def test_family_and_engineering_fixture_protection(tmp_path):
    text=(ROOT/'assets/navigation_original.map').read_text()
    reflected='\n'.join(row[::-1] for row in text.strip().splitlines())+'\n'
    (tmp_path/'reflected.map').write_text(reflected)
    data=json.loads(MANIFEST.read_text())
    data['layouts'][0]['path']=str(ROOT/'assets/navigation_original.map')
    data['layouts'].append(dict(id='heldout',family='fake-new-family',path='reflected.map',split='test',weight=1,sha256=geometry_hash(reflected)))
    path=tmp_path/'manifest.json';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='share a layout family'):load_layouts(path,split='test',mode='cross_layout')
    fixture=fixture_layout();(tmp_path/'fixture.map').write_text(fixture['layout'])
    data['layouts'][1].update(path='fixture.map',sha256=fixture['sha256'],engineering_fixture=True)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Engineering'):load_layouts(path,split='test',mode='cross_layout')


def test_launcher_stage_order_and_failure_gate(tmp_path,monkeypatch):
    import run_decentralized_navigation as launch
    import report_navigation
    stages=[]
    monkeypatch.setattr(launch,'run_validation',lambda *a: stages.append('validate') or tmp_path/'validated.json')
    monkeypatch.setattr(launch,'train_campaign',lambda *a:stages.append('train'))
    monkeypatch.setattr(launch,'evaluation_campaign',lambda *a:stages.append('evaluate'))
    monkeypatch.setattr(report_navigation,'generate_report',lambda *a:stages.append('report'))
    assert launch.main(['--output',str(tmp_path)])==0
    assert stages==['validate']
    stages.clear()
    assert launch.main(['--train','--output',str(tmp_path)])==0
    assert stages==['validate','train','evaluate','report']
    def fail(*a):raise ValueError('validation failed')
    monkeypatch.setattr(launch,'run_validation',fail);stages.clear()
    with pytest.raises(ValueError):launch.main(['--train','--output',str(tmp_path)])
    assert not stages
    assert json.loads((tmp_path/'status.json').read_text())['stage']=='failed'


def test_scheduler_reuses_completed_work_with_one_plus_three_gpu_split(tmp_path,monkeypatch):
    import run_decentralized_navigation as launch
    (tmp_path/'configs').mkdir()
    waves=[]
    probes=iter([[], [('0','GPU-A')]])
    monkeypatch.setattr(launch,'free_gpus',lambda:next(probes,[('0','GPU-A'),('1','GPU-B')]))
    waits=[]
    monkeypatch.setattr(launch.time,'sleep',lambda seconds:waits.append(seconds))
    monkeypatch.setattr(launch,'verify_training',lambda *a:None)
    def run(jobs):
        assert len(jobs)==4
        assert [j['physical_gpu'] for j in jobs]==['0','1','1','1']
        assert [j['environment']['CUDA_VISIBLE_DEVICES'] for j in jobs]==['GPU-A','GPU-B','GPU-B','GPU-B']
        assert len({j['seed'] for j in jobs})==1
        waves.append(jobs)
        for job in jobs:
            (Path(job['run_dir'])/'launch.json').write_text('{"exit_status":0}')
        return 0
    monkeypatch.setattr(launch,'run_wave',run)
    launch.train_campaign(tmp_path,MANIFEST,lambda *a:None,lambda:None)
    assert len(waves)==3 and waits==[30,30]
    assert [wave[0]['seed'] for wave in waves]==[0,1,2]
    launch.train_campaign(tmp_path,MANIFEST,lambda *a:None,lambda:None)
    assert len(waves)==3


def test_report_end_to_end_and_conservative_decisions(tmp_path,monkeypatch):
    import run_decentralized_navigation as launch
    from report_navigation import generate_report
    from navigation_train import CONDITIONS,sha,write_json
    original_suites=launch.suites
    monkeypatch.setattr(launch,'suites',lambda *args:[dict(s,count=2) for s in original_suites(*args)])
    env=episode()
    try:env.step(np.zeros(5,int));base=env.metrics()
    finally:env.close()
    for method,(communication,_) in CONDITIONS.items():
        for seed in range(3):
            folder=tmp_path/'evaluation'/method/f'seed_{seed}';folder.mkdir(parents=True)
            for kind in ('last','best'):
                for spec in launch.suites(communication,kind):
                    # All methods tie. This cannot demonstrate communication/CCPD benefit.
                    rows=[dict(base,steps=spec['steps'],requested_steps=spec['steps'],completed_cycles=10,
                               method=method,training_seed=seed,checkpoint_kind=kind,suite=spec['suite'],
                               episode_id=e,action_replicate=spec['replicate'],inference_ms_per_fleet_step=0.) for e in (2000,2001)]
                    path=folder/f"{kind}_{spec['suite']}_rng_{spec['replicate']}.jsonl"
                    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    write_json(path.with_name(path.name+'.done.json'),dict(inputs={},sha256=sha(path)))
    report=generate_report(tmp_path)
    assert report['complete'] and report['paired_comparisons']
    assert not any(d['promising'] for d in report['decisions'])
    assert 'not an equivalence claim' in (tmp_path/'next_steps.md').read_text()
    # Known favorable synthetic evidence passes; this is a report test, not a model result.
    for file in (tmp_path/'evaluation').rglob('*.jsonl'):
        records=[json.loads(line) for line in file.read_text().splitlines()]
        for row in records:
            row['completed_cycles']=11 if row['method'].startswith('communicating') else 10
            row['conflict_attempts']={'local':8,'local_ccpd':4,'communicating':4,'communicating_ccpd':2}[row['method']]
        file.write_text(''.join(json.dumps(row)+'\n' for row in records))
        write_json(file.with_name(file.name+'.done.json'),dict(inputs={},sha256=sha(file)))
    favorable=generate_report(tmp_path)
    assert all(d['promising'] for d in favorable['decisions'])
    path.unlink()
    with pytest.raises(ValueError,match='Incomplete'):generate_report(tmp_path)


def test_completion_at_gap_threshold_is_counted(monkeypatch):
    env=episode()
    actual=env.env.step
    def finish(actions):
        obs,rewards,term,trunc,info=actual(actions)
        info['completed_cycles'][0]=1
        info['cycle_time'][0]=500
        return obs,rewards,term,trunc,info
    monkeypatch.setattr(env.env,'step',finish)
    try:
        env.gap=499
        env.step(np.zeros(5,int))
        assert env.gap==0 and env.max_gap==500 and env.gap_events==1
        assert env.metrics()['fleet_failure']
    finally:env.close()
