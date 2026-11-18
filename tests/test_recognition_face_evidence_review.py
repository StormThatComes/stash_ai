import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace
import numpy as np
import pytest
from stash_ai.recognition.config import defaults
from stash_ai.recognition.face_evidence_review import apply, review, collapse, publish_proofs
from stash_ai.recognition.independent_evidence import count


def rows():
    geometry=[10,10,80,80,30,30,60,30,45,45,35,65,55,65,.99]
    return [dict(id=str(i),scan='s',cue=i,seconds=i,geometry=json.dumps(geometry),dimensions=2,
                 embedding=np.asarray([0,1] if i==2 else [1,0],dtype='<f4').tobytes(),
                 quality=json.dumps(dict(view_shot=0,transition_verified=True,alignment_valid=True,
                                         evidence_pixel_sha256=f'{i:064x}'))) for i in range(5)]


def test_anchor_demotion_requires_independent_bracketing_and_is_reversible():
    source=rows(); grouping=defaults()['clustering']
    assert set(review(source,grouping))=={'2'}
    assert all('query_only' not in json.loads(r['quality']) for r in source)
    source[2]['embedding']=source[0]['embedding']
    assert not review(source,grouping)
    source=rows()
    for r in source:r['quality']=json.dumps(dict(view_shot=0,transition_verified=False))
    assert not review(source,grouping)


def test_near_duplicate_votes_use_original_witness_without_transitive_drift():
    source=rows()[:3]
    pixels=np.tile(np.arange(32,dtype=np.uint8)*6,(32,3,1)).transpose(0,2,1)
    for i,r in enumerate(source):
        r['embedding']=source[0]['embedding']
        q=json.loads(r['quality']);q.update(evidence_thumbnail=(pixels+i*2).tobytes().hex(),evidence_aspect=1.)
        r['quality']=json.dumps(q)
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE observations(id TEXT,quality TEXT)')
        db.executemany('INSERT INTO observations VALUES (?,?)',[(r['id'],r['quality']) for r in source])
        collapse(db,source,dict(near_duplicate_pixel_rms=2.1,near_duplicate_similarity=.995))
        assert count(source)==2
        assert 'evidence_independence_unit' not in json.loads(source[2]['quality'])


@pytest.mark.parametrize('storage', ['disk', 'ram'])
@pytest.mark.parametrize('shared', [False, True], ids=['copied-vectors', 'shared-vectors'])
@pytest.mark.parametrize('kind', ['anchors', 'duplicates'])
def test_review_writes_private_metadata_and_preserves_shared_inputs(tmp_path, storage, shared, kind):
    from stash_ai.recognition.database import SCHEMA
    from stash_ai.recognition.rebuild import _copy_selection
    from stash_ai.recognition.replay_memory import memory_replay
    from stash_ai.recognition.replay_vector_inputs import install

    config = SimpleNamespace(data=defaults())
    config.data['execution']['rebuild_storage'] = storage
    config.data['clustering'].update(revalidate_face_anchors=kind == 'anchors',
                                    near_duplicate_pixel_rms=2.1 if kind == 'duplicates' else 0)
    evidence = rows() if kind == 'anchors' else rows()[:3]
    if kind == 'duplicates':
        pixels = np.tile(np.arange(32, dtype=np.uint8)*6, (32, 3, 1)).transpose(0, 2, 1)
        for i, row in enumerate(evidence):
            row['embedding'] = evidence[0]['embedding']
            quality = json.loads(row['quality'])
            quality.update(evidence_thumbnail=(pixels+i*2).tobytes().hex(), evidence_aspect=1.)
            row['quality'] = json.dumps(quality)
    path = tmp_path/'seed.sqlite3'
    selected = [dict(scene='1', scan='s')]
    with closing(sqlite3.connect(path, uri=True)) as source:
        source.row_factory = sqlite3.Row
        source.executescript(SCHEMA)
        source.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                       ('s', 'scope', '1', 'ns', 'input', 'policy', '{}', '{}', len(evidence), len(evidence), 'complete'))
        source.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
            [(r['id'], r['scan'], r['cue'], r['seconds'], 'sprite', '[]', r['geometry'],
              r['quality'], None, r['dimensions'], r['embedding'], None) for r in evidence])
        source.commit()
        before = path.read_bytes()
        if shared:
            source.execute('ATTACH DATABASE ? AS retained', (path.as_uri()+'?mode=ro&immutable=1',))
        # Separate trials must start from raw metadata, even after a reviewed
        # trial commits its private changes.
        for _ in range(2):
            with memory_replay(config) as (db, uri):
                _copy_selection(SimpleNamespace(connection=source), db, selected, 'scope', 'ns', bindings=False)
                view = db.execute("SELECT 1 FROM temp.sqlite_master WHERE name='observations' AND type='view'").fetchone()
                assert bool(view) is shared
                assert db.execute('SELECT COUNT(*) FROM main.observations WHERE embedding IS NOT NULL').fetchone()[0] == (0 if shared else len(evidence))
                original = [dict(r) for r in db.execute('SELECT * FROM observations ORDER BY id')]
                assert all('query_only' not in json.loads(r['quality']) and
                           'evidence_independence_unit' not in json.loads(r['quality']) for r in original)
                apply(db, selected, config)
                reviewed = [dict(r) for r in db.execute('SELECT * FROM observations ORDER BY id')]
                assert [(r['id'], r['embedding']) for r in reviewed] == [(r['id'], r['embedding']) for r in original]
                if kind == 'anchors':
                    quality = json.loads(reviewed[2]['quality'])
                    assert quality['query_only'] is True
                    assert quality['anchor_review']['kind'] == 'temporal_anchor_contradiction'
                    assert set(quality['anchor_review']['witness_ids']) == {'0', '1', '3', '4'}
                    assert count([r for r in reviewed if not json.loads(r['quality']).get('query_only')]) == 4
                    db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                               ('candidate', 's', 'policy', None, 'pending', json.dumps(dict(observation='2'))))
                    publish_proofs(db)
                    payload = json.loads(db.execute('SELECT payload FROM candidates').fetchone()[0])
                    assert payload['anchor_review'] == quality['anchor_review']
                else:
                    assert count(reviewed) == 2
                    assert json.loads(reviewed[0]['quality'])['evidence_independence_unit'] == json.loads(reviewed[1]['quality'])['evidence_independence_unit']
                    assert 'evidence_independence_unit' not in json.loads(reviewed[2]['quality'])
                db.commit()
                # A reopened reader reinstalls the same vector view and sees
                # committed review metadata from the private main table.
                with closing(sqlite3.connect(uri, uri=True)) as reader:
                    install(reader)
                    assert list(reader.execute('SELECT quality FROM observations ORDER BY id')) == [(r['quality'],) for r in reviewed]
            assert path.read_bytes() == before
