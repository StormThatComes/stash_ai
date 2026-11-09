from stash_ai.recognition.merge_solver_tighten import TRACK, route_parameters
from stash_ai.recognition.solver_validation import holdout_verdict
from stash_ai.recognition.sequence_validation import sequence_metrics


def test_local_track_is_credited_without_a_scene_bonus():
    counts = route_parameters([dict(reason='local_body_track',status='accepted',evidence_kind='person')])
    assert counts[TRACK] == 1
    assert 'sapiensid.identities.scene_match_bonus' not in counts


def test_contact_exit_loss_is_a_regression_even_with_correct_interior():
    labels, positions, samples = {}, {}, {}
    for p in 'AB':
        for i in range(3):
            oid = p+str(i)
            labels[oid], positions[oid] = p, ('1',i)
            samples[oid] = dict(slices=['entangled'] if i==1 else [])
    def measures(lost):
        decisions = {o:dict(accepted_identity=None if lost and o=='A2' else p,
                            evidence_kind='person') for o,p in labels.items()}
        return dict(sequences=sequence_metrics(labels,samples,positions,decisions),
                    partners=dict(false_joins=0,partner_pairs=2),
                    visible=dict(split_identities=0,reference_identities=2))
    verdict = holdout_verdict({'baseline/joint':measures(False),'winner/joint':measures(True)})
    assert verdict['status']=='regressed'
    assert verdict['measures']['lost_person_endpoints']['winner']==1


def test_rear_scene_with_tracks_keeps_core_timestamps(tmp_path):
    from types import SimpleNamespace
    from stash_ai.recognition import rear_recovery
    from stash_ai.recognition.body_identity import rebuild, policy
    from stash_ai.recognition.database import FaceDatabase
    from tests.test_recognition_body_identity import configuration, scene
    config = configuration(tmp_path)
    config.data['sapiensid']['identities']['tracklet_matching'] = True
    with FaceDatabase(config.database) as state:
        item, _ = scene(state.connection,config)
        state.connection.commit()
        rebuild(state,[item],config,{item['scan']:[]})
        decisions = rear_recovery._scene(state.connection,item['scan'],policy(config),
                                         SimpleNamespace(rank=lambda *args:[]),SimpleNamespace(block=8),config)
        assert decisions == {}


def test_export_mixed_holdout_counts(tmp_path):
    from benchmarks.benchmark_merge_solver import fixture
    from stash_ai.recognition.merge_goal_output import export
    from stash_ai.recognition.merge_report_policy import proposal
    config, results = fixture(tmp_path,'query')
    results.update(best_result=proposal(config,{}),simulation={},
                   holdout_validation=dict(verdict=dict(status='regressed',measures={
                       'lost': dict(baseline=0,winner=1,status='regressed'),
                       'returns': dict(baseline=1,winner=0,baseline_total=1,winner_total=1,status='regressed')})))
    dest = tmp_path/'proposal.yml'
    export(config,results,tmp_path/'result.json',dest,yaml_mode='diff')
    assert '0 -> 1 (regressed)' in dest.read_text() and '1/1 -> 0/1' in dest.read_text()


def test_fragment_replay_matches_full_scan_on_added_changed_and_removed_rival(tmp_path):
    from io import BytesIO
    from types import SimpleNamespace
    import numpy as np
    from stash_ai.recognition.database import FaceDatabase,encoded
    from stash_ai.recognition.derived_cache import scope
    from stash_ai.recognition.body_identity import policy
    from stash_ai.recognition.rear_gallery import Gallery
    from stash_ai.recognition.rear_pending import Pending
    from stash_ai.recognition.rear_fragments import select
    from stash_ai.recognition.rear_recovery import _scene
    from tests.test_recognition_body_identity import configuration, scene
    cfg = configuration(tmp_path)
    cfg.data['execution'].update(rebuild_cache_bytes=2**23,derived_cache_bytes=2**23)
    axis = np.zeros(4096,dtype='<f4')
    axis[0]=1.
    with FaceDatabase(cfg.database) as state, scope(state,cfg):
        db=state.connection
        item,ids=scene(db,cfg,vectors=[[1,0],[-1,0],[0,1]],times=[0,16,32])
        stamp = policy(cfg)
        db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.orientation','back')")
        for oid in ids:
            db.execute('INSERT INTO body_members VALUES (?,?,?,?,?,?)',(oid,None,'auxiliary','withheld',stamp,
                       encoded(dict(person=None,status='withheld',role='auxiliary',reason='body_anchor_quality'))))
        def gallery(owners):
            return Gallery([(owner,'external',[dict(id=owner+str(i),scan='external',roots=['l','r'],
                    quality='{}',_vector=v) for i in range(2)]) for owner,v in owners])
        scan=item['scan']
        for rival, expected in [(None,'A'),(axis,None),(-axis,'A'),(None,'A')]:
            owners=[('A',axis)]+([('B',rival)] if rival is not None else [])
            g=gallery(owners)
            from stash_ai.recognition.rear_gallery import revision
            g.revision=revision(g)
            search=SimpleNamespace(config=cfg,bank={p:dict(cues=set()) for p,_ in owners},unavailable=set(),
                                   mean_versions={},block=8,entries=[],index=BytesIO(),identities=[])
            pending=Pending(db,g,search,cfg.data['sapiensid']['identities'])
            blocks=select(db,scan,stamp,pending)
            assert blocks==([0,1] if rival is not None and rival[0]==-1 else [0])
            full=_scene(db,scan,stamp,g,search,cfg)
            partial=_scene(db,scan,stamp,g,search,cfg,blocks=blocks,pending=pending)
            assert encoded(full)==encoded(partial)
            assert partial.get(ids[0],{}).get('person')==expected
        db.execute("INSERT INTO people VALUES ('old','scope','ns',0,NULL,'old')")
        db.execute("UPDATE body_members SET person='old' WHERE observation IN (?,?)",ids[:2])
        assert select(db,scan,stamp,pending)==[0,1]
