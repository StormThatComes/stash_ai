import json
from dataclasses import replace
import pytest

from tests.test_recognition_merge_report import seeded
from stash_ai.recognition.cli import parser


@pytest.mark.parametrize('solver',[False,True])
def test_report_profile_covers_snapshot_and_final_publication(tmp_path,capsys,solver):
    config,_=seeded(tmp_path)
    selector=['--performer-set','900,901','--solve','--max-trials','1'] if solver else ['--performer-id','900','--performer-id','901']
    args=parser().parse_args(['report-merge',*selector,'--profile-report','--progress-style','off'])
    if solver:from stash_ai.recognition.merge_goals import execute
    else:from stash_ai.recognition.merge_report import execute
    assert execute(args,config)==0
    output=capsys.readouterr();report=json.loads(output.out)
    stored=report['report_performance']
    assert 'Copy retained snapshot' in stored['stages']
    assert 'before output serialization' in stored['scope']
    final=json.loads(next(line.split(': ',1)[1] for line in output.err.splitlines() if line.startswith('Merge report performance: ')))
    assert final['elapsed_seconds'] >= stored['elapsed_seconds']
    assert 'Serialize and publish report' in final['stages']
    if solver:assert {'Resolve goals','Fingerprint retained inputs','Solve merge goals'} <= final['stages'].keys()


def test_plain_pair_emits_stage_progress_without_contaminating_json(tmp_path,capsys):
    from stash_ai.recognition.merge_report import execute
    config,_=seeded(tmp_path)
    args=parser().parse_args(['report-merge','--performer-id','900','--performer-id','901'])
    assert execute(args,config)==0
    output=capsys.readouterr()
    assert json.loads(output.out)['kind']=='recognition_identity_merge'
    assert 'Copy retained snapshot' in output.err
    assert 'performance:' not in output.err


def test_failed_trial_retains_partial_metrics_and_resource_dimensions(tmp_path):
    from stash_ai.recognition.trial_evaluation import evaluate
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.resource_errors import ResourceLimit
    config,_=seeded(tmp_path);data=plain(config.data);data['execution']['profile_rebuild']=True
    config=replace(config,data=data)
    error=ResourceLimit('synthetic plan limit',budget='sapiensid.execution.plan_bytes',limit=100,requested=101)
    def fail(*args):raise error
    with pytest.raises(ResourceLimit) as caught:
        evaluate('trial-1',config,seed=None,source=None,scope='scope',namespace='ns',installation='test',
                 scans=[],observations={},rebuild=fail,copy_inputs=lambda *args:None)
    assert caught.value is error
    assert {'Prepare replay','Rebuild selected identities'} <= error.trial_profile['stages'].keys()
    assert error.document()==dict(kind='configured_resource_limit',budget='sapiensid.execution.plan_bytes',limit=100,requested=101)
    assert error.trial_profile['elapsed_seconds'] >= 0


def test_feasibility_failures_exclude_other_search_phases():
    from stash_ai.recognition.merge_solver_feasibility import Feasibility
    f=Feasibility.__new__(Feasibility);f.groups=[]
    result=f.report({},dict(a=dict(search_phase='adaptive'),b=dict(search_phase='feasibility')),6)
    assert result['probes_failed']==1 and set(result['failed_probes'])=={'b'}
