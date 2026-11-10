from stash_ai.recognition.solver_validation import _metrics
from stash_ai.recognition.crowd_labels import detection_counts,validate


def test_joint_id_cannot_recover_turn_and_duplicate_projection_does_not_add_a_person():
    labels,locations,samples,results = {},{},{},{}
    for cue in range(3):
        for truth in 'AB':
            for projection in (['face','body'] if cue!=1 else ['body']):
                oid = f'{truth}-{cue}-{projection}'
                labels[oid] = truth
                locations[oid] = '1',cue
                samples[oid] = dict(slices=['back','entangled'] if cue==1 else [])
                results[oid] = dict(accepted_identity='X',evidence_kind='person' if projection=='body' else 'face',
                                    evidence=dict(reason='known_identity'))
    values = _metrics(labels,samples,locations,results)
    assert values['physical']['observations']==6
    assert values['sequences']['one_id_turns']==2
    assert values['sequences']['own_id_turns']==0 and values['sequences']['anchored_turns']==2
    results['A-0-body']['accepted_identity']='foreign'
    assert _metrics(labels,samples,locations,results)['physical']['projection_conflicts']==1


def test_visible_partner_can_be_missing_without_an_observation_id(tmp_path):
    from tests.test_recognition_body_identity import configuration,scene
    from tests.test_recognition_crowded_tracks import pose
    from stash_ai.recognition.database import FaceDatabase,encoded
    cfg = configuration(tmp_path)
    visible = {'1':{'0':[dict(truth='A',point=[40,45]),dict(truth='B',point=[110,45])]}}
    validate(visible,['1'])
    with FaceDatabase(cfg.database) as state:
        item,ids = scene(state.connection,cfg)
        state.connection.execute('UPDATE body_observations SET geometry=? WHERE id=?',(encoded(pose(40)),ids[0]))
        values = detection_counts(state.connection,[dict(scene='1',id=item['scan'],total=3)],visible)
        assert (values['expected_people'],values['missing_poses'],values['duplicate_poses'])==(2,1,0)


def test_face_after_cut_is_not_a_local_turn_anchor():
    from stash_ai.recognition.sequence_validation import sequence_metrics
    positions = {'start':10,'back':11,'end':12,'far':100}
    ids = {k:'A' for k in positions}
    decisions = {k:dict(accepted_identity='A',evidence_kind='person') for k in ids}
    decisions['far'].update(evidence_kind='face',evidence=dict(reason='known_identity'))
    result = sequence_metrics(ids,{k:dict(slices=['back'] if k=='back' else []) for k in ids},
                              {k:('1',v) for k,v in positions.items()},decisions)
    assert (result['anchored_turns'],result['remote_root_turns'])==(0,1)


def test_visible_people_have_one_outcome_even_when_a_detection_is_missing():
    from stash_ai.recognition.contact_holdout import visible_scores
    groups = {('s',0,'A'):['root'],('s',1,'A'):['back'],('s',1,'B'):[],('s',2,'A'):['new']}
    decisions = dict(root=dict(accepted_identity='X',evidence={'reason':'known_identity'}),
                     back=dict(accepted_identity=None),new=dict(accepted_identity='Y'))
    counts = visible_scores(groups,decisions,{'root':'A'})
    assert counts==dict(total=4,reference_identities=1,split_identities=0,correct=1,partner=0,fragment=1,withheld=1,missing=1,unknown_reference=0,split_reference=0)


def test_private_association_fault_swaps_actual_projections(tmp_path):
    import numpy as np
    from tests.test_recognition_body_identity import configuration,scene
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.contact_holdout import swap
    from stash_ai.recognition.person_signal import read,contract
    cfg = configuration(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        item,ids = scene(db,cfg,vectors=[[1,0],[0,1]],times=[0,0])
        for face,body in zip('ab',ids):
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',(face,item['scan'],0,0.,'','[]','[]','{}',None,2,np.array([1,0],dtype='<f4').tobytes(),None))
            db.execute('UPDATE body_observations SET face=? WHERE id=?',(face,body))
        swap(db,cfg,item['scan'],0,*ids)
        assert db.execute('SELECT face FROM body_observations WHERE id=?',(ids[1],)).fetchone()[0]=='a'
        assert np.frombuffer(read(db,'a',contract(cfg))['embedding'],dtype='<f4')[:2].tolist()==[0,1]


def test_association_fault_counts_changed_owners_without_inflating_later_cues():
    from stash_ai.recognition.contact_holdout import spread
    locations = dict(face_a=('s',0),face_b=('s',0),contact=('s',1),later=('s',2),external=('other',0))
    labels = dict(face_a='A',face_b='B',contact='A',later='A',external='A')
    before = {k:dict(accepted_identity=p,evidence_kind='face' if k.startswith('face') else 'person',
                     evidence=dict(reason='known_identity')) for k,p in labels.items()}
    after = {k:dict(v) for k,v in before.items()}
    after['contact']['accepted_identity']='B'
    report = spread(before,after,locations,'s',1,labels)
    assert report['changed_observations']==report['changed_cues']==1
    assert report['later_cues']==0 and report['other_scenes']==[]
    assert report['partition_changed_cues']>1


def test_unbound_manual_people_remain_in_physical_denominator():
    from stash_ai.recognition.contact_holdout import visible_scores
    labels={'root':'A'}
    samples={'root':dict(slices=[])}
    decisions={'root':dict(accepted_identity='owner',evidence=dict(reason='known_identity'))}
    omitted={('1',1,'B'):dict(binding='missing',slices=['back']),('1',2,'A'):dict(binding='ambiguous')}
    physical=_metrics(labels,samples,{'root':('1',0)},decisions,unbound=omitted)['physical']
    assert physical['coverage']==1/3 and physical['slices']['back']['withheld']==1
    groups={key:[] for key in omitted}
    groups['1',0,'A']=['root']
    scores=visible_scores(groups,decisions,labels,bindings=omitted)
    assert (scores['correct'],scores['missing'],scores['ambiguous'])==(1,1,1)
