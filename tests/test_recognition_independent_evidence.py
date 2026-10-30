import json
import numpy as np
from stash_ai.recognition.independent_evidence import unit
from stash_ai.recognition.query_support import cohorts
from tests.test_recognition_clustering import setup


def test_repeated_pixels_cannot_supply_multiple_query_votes(tmp_path):
    config=setup(tmp_path)
    rows=[dict(id=str(i),scan='s',seconds=i,dimensions=4,
        embedding=np.array([1,0,0,0],dtype='<f4').tobytes(),
        quality=json.dumps(dict(evidence_crop_sha256='a'*64))) for i in range(4)]
    assert len({unit(r) for r in rows})==1
    assert not list(cohorts(rows,config,key=lambda r:'owner',prototype=lambda r:'view',minimum=lambda _:2))
    rows[-1]['quality']=json.dumps(dict(evidence_crop_sha256='b'*64))
    assert list(cohorts(rows,config,key=lambda r:'owner',prototype=lambda r:'view',minimum=lambda _:2))


def test_equal_vectors_without_pixel_proof_remain_independent():
    assert unit(dict(scan='s',seconds=1)) != unit(dict(scan='s',seconds=2))
    assert unit(dict(scan='s',seconds=1,quality='{}')) == unit(dict(scan='s',seconds=1))


def test_body_goal_votes_use_cues_and_still_deduplicate_pixel_copies():
    from stash_ai.recognition.independent_evidence import count,representatives
    from stash_ai.recognition.merge_goal_identity import units
    rows=[dict(id=str(i),scan=s,scene=s,cue=cue,seconds=0,evidence_kind='person',
               quality=json.dumps({'evidence_pixel_sha256':pixel*64}))
          for i,(s,cue,pixel) in enumerate([('s',0,'a'),('s',0,'b'),('s',1,'c'),
                                            ('s',2,'a'),('t',0,'d'),('u',0,'c')])]
    assert count(rows)==4
    assert len(representatives(rows))==len(units(rows))==4
    for i,row in enumerate(rows):row['seconds']=i*30
    assert count(rows)==4
    assert count([dict(scan='s',cue=i,seconds=0,evidence_kind='person') for i in range(3)])==3


def test_votes_require_both_distinct_times_and_pixels_in_every_order():
    from itertools import permutations
    from stash_ai.recognition.independent_evidence import count
    rows = [dict(scan='s',seconds=t,quality=json.dumps({'evidence_crop_sha256':p*64}))
            for t,p in [(1,'a'),(2,'a'),(3,'b'),(3,'c')]]
    assert {count(order) for order in permutations(rows)} == {2}
    assert count([dict(scan='s',seconds=1),dict(scan='s',seconds=1)]) == 1


def test_duplicate_can_improve_representative_without_extra_vote():
    from stash_ai.recognition.view_prototypes import _Cluster
    slot=_Cluster('other','first')
    sample=np.asarray([1.,0.],dtype='<f4')
    base=dict(scan='s',seconds=1,quality=json.dumps(dict(pixels=32,sharpness=1)))
    slot.add(dict(base,id='first'),sample)
    slot.add(dict(base,id='better',quality=json.dumps(dict(pixels=256,sharpness=100))),sample)
    assert slot.observations == ['better']
    assert len(slot.cues) == 1
    assert np.array_equal(slot.total,sample)


def test_raw_fingerprint_deduplicates_copies_across_scenes():
    from stash_ai.recognition.independent_evidence import count
    rows = [dict(scan=s, seconds=t, quality=json.dumps(dict(evidence_pixel_sha256='a'*64)))
            for s,t in [('first', 0), ('copy', 30)]]
    assert count(rows) == 1


def test_near_copy_requires_pixels_aspect_texture_and_embedding():
    from stash_ai.recognition.evidence_pixels import verified
    rng = np.random.default_rng(7)
    pixels = rng.integers(10, 245, 3072, dtype='uint8')
    a = np.array([1.,0.],dtype='<f4')
    assert verified(pixels, pixels+1, a, a, 1., 1.)
    assert not verified(pixels, pixels+10, a, a, 1., 1.)
    assert not verified(pixels, pixels, a, a[::-1], 1., 1.)
    assert not verified(pixels, pixels, a, a, 1., 0.)
    assert not verified(np.zeros(3072), np.zeros(3072), a, a, 1., 1.)


def test_pixel_fingerprint_does_not_depend_on_jpeg_budget():
    from PIL import Image
    from stash_ai.recognition.quality import crop_bytes
    from types import SimpleNamespace
    image = Image.fromarray(np.random.default_rng(3).integers(0,256,(128,128,3),dtype='uint8'))
    first, second = {}, {}
    face = SimpleNamespace(box=(10,10,100,100))
    try:
        a,_ = crop_bytes(image,face,limit=100000,details=first)
        b,_ = crop_bytes(image,face,limit=1000,details=second)
        assert a != b
        assert first['evidence_pixel_sha256'] == second['evidence_pixel_sha256']
    finally:
        image.close()


def test_compression_aliases_use_mutable_replay_metadata_with_run_cache(tmp_path):
    import sqlite3
    from stash_ai.recognition.evidence_pixels import canonicalize
    from stash_ai.recognition.run_cache import scope
    pixels=np.random.default_rng(4).integers(10,245,3072,dtype='uint8')
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
    try:
        db.execute('CREATE TABLE observations(id TEXT,quality TEXT,embedding BLOB,rejected TEXT)')
        for identifier,values in [('a',pixels),('b',pixels+1)]:
            db.execute('INSERT INTO observations VALUES (?,?,?,NULL)',(identifier,json.dumps(dict(
                evidence_pixel_sha256=identifier*64,evidence_thumbnail=values.tobytes().hex(),evidence_aspect=1.)),
                np.array([1.,0.],dtype='<f4').tobytes()))
        with scope(setup(tmp_path)):
            canonicalize(db)
        assert {json.loads(row[0])['evidence_pixel_sha256'] for row in db.execute('SELECT quality FROM observations')}=={'a'*64}
    finally:
        db.close()


def test_alias_scope_matches_replay_without_writing_source(tmp_path):
    import sqlite3
    from stash_ai.recognition.evidence_pixels import alias_scope,canonicalize
    from stash_ai.recognition.independent_evidence import count
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
    pixels=np.random.default_rng(4).integers(10,245,3072,dtype='uint8')
    try:
        db.execute('CREATE TABLE observations(id TEXT,scan TEXT,seconds REAL,quality TEXT,embedding BLOB,rejected TEXT)')
        for i in range(3):
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,NULL)',(str(i),'s',i,json.dumps(dict(
                evidence_pixel_sha256=str(i)*64,evidence_thumbnail=(pixels+i).tobytes().hex(),evidence_aspect=1.)),
                np.array([1.,0.],dtype='<f4').tobytes()))
        db.commit();before=db.total_changes
        rows=lambda:list(db.execute('SELECT * FROM observations'))
        assert count(rows())==3
        db.execute('PRAGMA query_only=ON')
        with alias_scope(db):
            assert count(rows())==1
            with alias_scope(db):assert count(rows())==1
        assert db.total_changes==before and count(rows())==3
        db.execute('PRAGMA query_only=OFF')
        canonicalize(db)
        assert count(rows())==1
    finally:db.close()


def test_face_compression_aliases_do_not_deduplicate_body_votes(tmp_path):
    from tests.test_recognition_merge_report import seeded
    from tests.test_recognition_hybrid_recovery import enable
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.hybrid_evidence import catalogue
    config,_=seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;config=enable(config,db)
        pixels=np.random.default_rng(4).integers(10,245,3072,dtype='uint8')
        rows=list(db.execute('SELECT id,quality FROM observations ORDER BY id'))
        for i,row in enumerate(rows):
            quality=json.loads(row['quality']);quality.update(evidence_pixel_sha256=f'{i:064x}',
                evidence_thumbnail=(pixels+i%2).tobytes().hex(),evidence_aspect=1.)
            db.execute('UPDATE observations SET quality=? WHERE id=?',(json.dumps(quality),row['id']))
        db.commit();before=db.total_changes
        ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        identities=catalogue(db,config,'scope',ns)
        assert all(len(i.independent)==3 for i in identities)
        assert db.total_changes==before
