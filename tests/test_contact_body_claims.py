from types import SimpleNamespace
import json
import numpy as np

from stash_ai.recognition.body_identity import defaults
from stash_ai.recognition.body_tracklets import paths, split_members
from stash_ai.recognition.contact_anchors import questionable_links, joint_body_claims
from tests.test_recognition_crowded_tracks import pose


def body(i,cue,x,owner=None,**quality):
    values = dict(detected_people=1,overlap=0.,frame_signature='11'*48,view_shot=0,pixel_digest=i)
    values.update(quality)
    return dict(id=i,cue=cue,seconds=cue,scan='s',face_person=owner,face_anchor=bool(owner),
                quality=json.dumps({'quality':values}),geometry=json.dumps(pose(x)),_vector=np.array([1.,0.]))


def test_contact_root_compares_both_sides():
    rows = [body('left',0,50,'A'),body('contact',1,50,'A',detected_people=2,overlap=.4),body('right',2,50,'B')]
    assert questionable_links(rows,defaults(),SimpleNamespace(face_similarity=np.dot))=={'contact'}


def test_partial_pair_preserves_only_the_separated_torso():
    rows = [body('A',0,30,detected_people=2),body('B',0,150,detected_people=2),
            body('a',1,30,expected_people=2)]
    chains = list(paths(rows,defaults(),SimpleNamespace(face_similarity=np.dot)))
    assert [[r['id'] for r in group] for group,_ in chains]==[['A','a']]


def test_two_interval_split_keeps_unknown_middle():
    rows = [body(str(i),i,30) for i in range(5)]
    proposed = {'0':'A','1':'A','3':'B','4':'B'}
    assert split_members(rows,proposed,lambda r,p:(r['id'],p) in {('1','B'),('3','A')},defaults())==set(proposed)
    assert not split_members(rows,proposed,lambda *args:False,defaults())


def test_joint_contact_explains_second_winner_with_the_other_body():
    rows=[body('x',0,30),body('y',0,140)]
    d={r['id']:dict(person=None,status='withheld') for r in rows}
    options={r['id']:[(v,v,p,[r],True,v) for p,v in zip('AB',scores)]
             for r,scores in zip(rows,[(.99,.86),(.94,.93)])}
    joint_body_claims(rows,d,options,defaults())
    assert [d[r['id']]['person'] for r in rows]==['A','B']


def test_joint_resolution_also_uses_faceless_scene_banks_with_bonus_disabled(tmp_path):
    from stash_ai.recognition.body_identity import Scores
    from stash_ai.recognition.scene_fragments import attach_with_scene_support
    from tests.test_recognition_body_identity import configuration
    cfg=configuration(tmp_path)
    settings=cfg.data['sapiensid']['identities']
    settings.update(scene_match_bonus=0.,ambiguity_margin=.04,match_threshold=.85,cohesion_threshold=.8)
    def sample(oid,cue,x,angle):
        row=body(oid,cue,x)
        radians=np.deg2rad(angle)
        row['_vector']=np.array([np.cos(radians),np.sin(radians)],dtype='float32')
        return row
    groups=[[sample(p+str(i),i,x,angle) for i in range(3)] for p,x,angle in [('A',30,0),('B',150,26)]]
    queries=[sample('x',3,30,0),sample('y',3,150,12)]
    rows=groups[0]+groups[1]+queries
    decisions={r['id']:dict(person=r['id'][0],status='accepted',role='anchor',reason='body_scene_core') for r in rows[:-2]}
    for r in queries:
        decisions[r['id']]=dict(person='A',status='accepted',role='auxiliary',reason='body_core_attachment')
    attach_with_scene_support(rows,groups,{},decisions,cfg,Scores(cfg),SimpleNamespace(allows=lambda *a:True))
    assert [decisions[r['id']]['person'] for r in queries]==['A','B']


def test_incomplete_peer_must_be_physically_excluded_and_empty_cue_must_be_empty():
    scores = SimpleNamespace(face_similarity=np.dot)
    def chains(rows,occupied=()):
        return [[r['id'] for r in chain] for chain,_ in paths(rows,defaults(),scores,occupied)]
    pair = [body('A',0,30,detected_people=2),body('a',1,30,detected_people=2)]
    peers = [body('B',0,160,frame_signature=None),body('b',1,160,frame_signature=None)]
    assert chains(pair+peers)==[['A','a']]
    peers[0]['geometry'] = json.dumps(pose(30))
    assert not chains(pair+peers)
    gap = [body('A',0,30),body('a',2,30)]
    assert chains(gap)==[['A','a']]
    assert not chains(gap,[1])


def test_pan_compensation_confirms_crowded_face():
    before = np.random.default_rng(7).integers(0,255,(4,4,3),dtype=np.uint8)
    after = np.roll(before,1,axis=1)
    rows = [body('a',0,40,'A',frame_size=[400,200],frame_signature=before.tobytes().hex()),
            body('b',1,140,'A',frame_size=[400,200],frame_signature=after.tobytes().hex(),detected_people=2,overlap=.3)]
    assert not questionable_links(rows,defaults(),SimpleNamespace(face_similarity=np.dot))
