import json
from types import SimpleNamespace
import numpy as np
from stash_ai.recognition.quality import alignment_measurements
from stash_ai.recognition.reference_selection import quality
from stash_ai.recognition.visibility import CONTRACT_ID


def test_visibility_and_alignment_reduce_quality_without_rejecting_old_evidence():
    raw = dict(pixels=200, sharpness=400, pose_deviation=.1)
    baseline = quality(json.dumps(raw))
    assert baseline > 0
    assert quality(json.dumps(dict(raw, alignment_error=.1))) < baseline
    assert quality(json.dumps(dict(raw, visible_face_fraction=.4,
                                  visibility=dict(contract=CONTRACT_ID)))) < baseline
    assert quality(json.dumps(dict(raw, alignment_valid=False))) == 0


def test_alignment_residual_is_invariant_to_source_scale_and_translation():
    landmarks=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                        [41.5493,92.3655],[70.7299,92.2041]])
    first=alignment_measurements(SimpleNamespace(landmarks=landmarks))
    second=alignment_measurements(SimpleNamespace(landmarks=landmarks*2+15))
    assert first['alignment_error'] < 1e-6 and second['alignment_error'] < 1e-6
    assert abs(second['alignment_scale']-.5) < 1e-6
    landmarks[2,0]+=12
    assert alignment_measurements(SimpleNamespace(landmarks=landmarks))['alignment_error'] > .01


def test_supported_rare_mode_has_reserved_slot():
    from stash_ai.recognition.reference_selection import select
    def row(i, vector, pixels):
        return dict(id=str(i),scan='s',scene='s',seconds=i,geometry='[]',dimensions=2,
                    embedding=np.array(vector,dtype='<f4').tobytes(),
                    quality=json.dumps(dict(pixels=pixels,sharpness=pixels)))
    rows=[row(i,[1.,0.],500) for i in range(30)]
    rows += [row(i,[.8,.6],50) for i in range(30,33)]
    chosen=select(rows,2)
    assert len(chosen)==2 and any(float(v[1])>.5 for _,v in chosen)


def test_seed_replacement_keeps_label_origin_and_is_one_time(tmp_path):
    from tests.test_recognition_clustering import setup, scan
    from stash_ai.recognition.clustering import Group, vector
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.reference_selection import refresh_local
    from stash_ai.recognition.reference_usage import ReferenceUsage
    config=setup(tmp_path)
    with FaceDatabase(config.database) as state:
        sid,_=scan(state,config,'1',count=5)
        rows=list(state.connection.execute('SELECT * FROM observations WHERE scan=? ORDER BY cue',(sid,)))
        state.connection.execute('UPDATE observations SET quality=? WHERE id=?',
                                 (json.dumps(dict(pixels=1,sharpness=0)),rows[0]['id']))
        group=Group('stable-label',[vector(rows[0])],False,(rows[0]['id'],))
        usage=ReferenceUsage(10000);usage.add(group.samples,kind='seed')
        for row in rows[1:]:
            refresh_local(state.connection,group,row,5,usage,config=config)
        assert group.identifier=='stable-label'
        assert group._identity_origin==rows[0]['id']
        assert group._seed_validated and group.sample_ids[0]!=rows[0]['id']
        protected=group.sample_ids[0]
        refresh_local(state.connection,group,rows[-1],5,usage,config=config)
        assert group.sample_ids[0]==protected


def test_late_supported_mode_displaces_unconfirmed_early_modes():
    from stash_ai.recognition.reference_selection import select
    def row(i, angle, pixels):
        return dict(id=str(i),scan='s',scene='s',seconds=i,geometry='[]',dimensions=2,
            embedding=np.array([np.cos(angle),np.sin(angle)],dtype='<f4').tobytes(),
            quality=json.dumps(dict(pixels=pixels,sharpness=pixels)))
    rows=[row(0,0,500),row(1,np.pi/3,500)]
    rows += [row(i,np.pi/6,50) for i in range(2,5)]
    assert any(i in ('2','3','4') for i,_ in select(rows,2))


def test_repeated_pixels_across_scenes_take_one_reference_slot():
    from stash_ai.recognition.reference_selection import select
    rows=[dict(id=str(i),scan=str(i),scene=str(i),seconds=i,geometry='[]',dimensions=2,
        embedding=np.array([1.,0.],dtype='<f4').tobytes(),
        quality=json.dumps(dict(pixels=50+i,sharpness=50,evidence_pixel_sha256='a'*64))) for i in range(3)]
    assert [i for i,_ in select(rows,3)] == ['2']


def test_better_duplicate_returns_after_exemplar_eviction_without_new_vote():
    from stash_ai.recognition.view_prototypes import _Cluster
    def row(i,t,q):return dict(id=i,scan='s',seconds=t,quality=json.dumps(dict(pixels=q,sharpness=q)))
    slot=_Cluster('other','old')
    slot.add(row('old',0,1),np.array([1.,0.]),maximum=1)
    slot.add(row('other',1,10),np.array([1.,0.]),maximum=1)
    assert 'old' not in slot.observations
    total=slot.total.copy()
    slot.add(row('better',0,500),np.array([1.,0.]),maximum=1)
    assert slot.observations == ['better']
    assert len(slot.cues)==2 and slot.centre_count==2
    assert np.array_equal(total,slot.total)


def test_late_high_quality_mode_can_replace_saturated_confirmed_reserves():
    from stash_ai.recognition.reference_selection import select
    rows=[]
    for mode in range(5):
        sample=np.zeros(6,dtype='<f4');sample[0]=np.sqrt(.7);sample[mode+1]=np.sqrt(.3)
        for j in range(3 if mode<4 else 20):
            rows.append(dict(id=f'{mode}:{j}',scan='s',scene='1',seconds=len(rows),geometry='[]',
                dimensions=6,embedding=sample.tobytes(),quality=json.dumps(dict(
                    pixels=80 if mode<4 else 500,sharpness=30 if mode<4 else 900,pose_deviation=.1))))
    chosen=select(rows,2)
    assert len(chosen)==2
    assert any(identifier.startswith('4:') for identifier,_ in chosen)
    assert [i for i,_ in chosen]==[i for i,_ in select(rows,2)]
