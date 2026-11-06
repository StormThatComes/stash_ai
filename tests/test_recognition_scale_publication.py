import json
import pytest
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.publication_probe import Probe
from stash_ai.recognition.publication_batch import ensure_people,publish_people
from stash_ai.runtime.report import RunReport
from tests.test_recognition_publication_batch import Batched,bind
from tests.test_recognition_merge_report import seeded


def test_probe_reuses_batch_without_writes_and_detects_external_deletion(tmp_path):
    config,people=seeded(tmp_path);client=Batched();report=RunReport(False)
    with FaceDatabase(config.database) as state:
        bind(state,client,people)
        for scene in ('1','2'):
            targets=ensure_people(state,client,report,'scope',scene,people)
            publish_people(state,client,report,'scope',scene,targets)
        namespace=state.connection.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        probe=Probe(state,client,'scope',namespace)
        items=[dict(scene=s,scan=s) for s in ('1','2')]
        result={s:[dict(person=p,status='accepted') for p in people] for s in ('1','2')}
        client.batch_reads.clear();writes=list(client.batch_writes)
        probe.prepare(items,result)
        assert probe.unchanged('1',people)==targets and probe.unchanged('2',people)==targets
        assert len(client.batch_reads)==1 and client.batch_writes==writes
        client.links['2'].remove(next(iter(targets.values())))
        probe.prepare(items,result)
        assert probe.unchanged('1',people)==targets and probe.unchanged('2',people) is None
        state.connection.execute("UPDATE publications SET status='unknown' WHERE scene='1'")
        assert probe.unchanged('1',people) is None
        probe.clear();assert probe.unchanged('2',people) is None


def test_warm_run_uses_publication_probe_and_still_checks_ownership(tmp_path,monkeypatch):
    from tests.test_recognition_run import setup,run
    config,client,services,_=setup(tmp_path,monkeypatch)
    assert run(config,services,'1','2')==0
    assert run(config,services,'1','2')==0
    with FaceDatabase(config.database) as state:
        summary=json.loads(state.connection.execute('SELECT summary FROM runs ORDER BY rowid DESC').fetchone()[0])
        assert summary['counts']['scenes_publication_reused']==2
    performer=client.links['1'][-1];client.profiles[performer]['details']='manual edit'
    assert run(config,services,'1')==1


def test_scene_ranking_composes_unchanged_scene_after_another_changes(tmp_path,monkeypatch):
    from tests.test_recognition_vtt_portraits import setup
    from tests.test_recognition_run import run
    from stash_ai.recognition import vtt_portraits
    from stash_ai.recognition.vtt_ranking_cache import Ranking
    from stash_ai.recognition.portrait_refresh import retained_targets
    from stash_ai.recognition.clustering import _groups,reference_context
    from stash_ai.recognition.portraits import policy_identity
    config,client,services,_=setup(tmp_path,monkeypatch)
    assert run(config,services,'1','2')==0
    with FaceDatabase(config.database) as state:
        person=state.connection.execute('SELECT * FROM people WHERE performer IS NOT NULL').fetchone()
        target=next(iter(retained_targets(state,config,person['scope'],['1','2']).values()))
        context=reference_context(_groups(state.connection,person['scope'],person['namespace'],config.data['clustering']))
        ranking=Ranking(state.connection,config,person['scope'],target,person['namespace'],context,policy_identity(config))
        # Establish per-scene signatures independently from the run-captured matching proof.
        ranks=[]
        for row in state.connection.execute("SELECT id,scene,metadata FROM portraits WHERE json_extract(metadata,'$.score') IS NOT NULL"):
            m=json.loads(row['metadata']);ranks.append((-m['score'],-m['quality_tiebreak'],row['scene'],m['cue_seconds'],row['id']))
        ranking.save_scenes(ranks)
        assert len(ranking.load_scenes())==2
        dirty=next(i['scan'] for i in target['scans'] if i['scene']=='2')
        state.connection.execute("UPDATE observations SET quality=json_set(quality,'$.sharpness',0) WHERE scan=?",(dirty,))
        available=ranking.load_scenes()
        assert set(available)=={i['scan'] for i in target['scans'] if i['scene']=='1'}
