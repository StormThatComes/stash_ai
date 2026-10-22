"""Additional valid queries preserve independently supported appearance coverage."""
import numpy as np
import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_corroborated_views import bank, seed
from tests.test_recognition_deterministic_resolution import configured
from tests.test_recognition_frozen_partition import fixture
from tests.test_recognition_query_support import appearance
from tests.test_recognition_scene_context import candidates


def promoted(db, config):
    return [v for v in bank(db, config).views if v.authority == 'corroborated_view']


def assert_owner(db, config, item, owner, *, attachment=False):
    rows = candidates(db, item['scan'], config)
    assert rows and all(s == 'accepted' and p == owner for s, p, _ in rows)
    if attachment:
        assert all(d['reference_role'] == 'attachment' for _, _, d in rows)


@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('full_replay', [False, True])
def test_another_root_pair_cannot_erase_an_existing_supported_appearance(tmp_path, sign, full_replay):
    config = configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source = fixture(db, config, '1', [appearance(0)] * 3 +
            [appearance(45, -40)] * 3 + [appearance(45)] * 3 +
            [appearance(45, 40)] * 3, range(3, 12))
        rebuild_selected(db, [source], config)
        owner = candidates(db, source['scan'], config)[0][1]
        queries = []
        for scene in ('2', '3'):
            item = fixture(db, config, scene, [appearance(65, sign * 6)] * 3)
            rebuild_selected(db, [item], config)
            queries.append(item)
        assert promoted(db, config)
        extra = fixture(db, config, '4', [appearance(65, -sign * 6)] * 3)
        rebuild_selected(db, [extra], config)
        views = promoted(db, config)
        assert any(np.dot(v.group.samples[0], appearance(65, sign * 6)) > .9999 for v in views)
        future = fixture(db, config, '5', [appearance(95, sign * 6)] * 3)
        selected = [source, *queries, extra, future] if full_replay else [future]
        rebuild_selected(db, selected, config)
        for item in (source, *queries, extra, future):
            assert_owner(db, config, item, owner, attachment=item is future)
        assert db.connection.execute('SELECT COUNT(DISTINCT person) FROM members').fetchone()[0] == 1


@pytest.mark.parametrize('extra_votes', [12, 40])
@pytest.mark.parametrize('full_replay', [False, True])
def test_a_more_frequent_look_preserves_a_supported_edge_representative(tmp_path, extra_votes, full_replay):
    config = configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source, queries = seed(db, config)
        owner = candidates(db, source['scan'], config)[0][1]
        extra = fixture(db, config, '4', [appearance(50)] * extra_votes)
        rebuild_selected(db, [extra], config)
        views = promoted(db, config)
        assert any(np.dot(v.group.samples[0], appearance(65)) > .9999 for v in views)
        assert any(np.dot(v.group.samples[0], appearance(50)) > .9999 for v in views)
        for view in views:
            proof = view.promotion
            assert proof['query_support'] >= proof['required_support']
            assert proof['scene_count'] >= 2 and proof['generation'] == 1
            assert not set(proof['parents'][0]['support_ids']).intersection(proof['parents'][1]['support_ids'])
        future = fixture(db, config, '5', [appearance(97)] * 3)
        selected = [source, *queries, extra, future] if full_replay else [future]
        rebuild_selected(db, selected, config)
        assert_owner(db, config, future, owner, attachment=True)
        before = [tuple(r) for r in db.connection.execute('SELECT * FROM members ORDER BY observation')]
        rebuild_selected(db, selected, config)
        assert before == [tuple(r) for r in db.connection.execute('SELECT * FROM members ORDER BY observation')]


def test_representatives_with_identical_vectors_do_not_duplicate_the_query_bank(tmp_path):
    config = configured(tmp_path)
    with FaceDatabase(config.database) as db:
        seed(db, config, votes=12)
        views = promoted(db, config)
        assert len(views) == 1
        assert views[0].promotion['query_support'] == 24
