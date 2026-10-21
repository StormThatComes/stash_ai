"""Independent supported looks survive medoid frequency changes."""
import numpy as np
import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.global_views import catalogue
from stash_ai.recognition.reference_usage import ReferenceUsage
from tests.test_recognition_anchor_views import config_for,A,X,C
from tests.test_recognition_frozen_partition import fixture
from tests.test_recognition_reference_registry import angle
from tests.test_recognition_rebuild import changed
from tests.test_recognition_scene_context import candidates


@pytest.mark.parametrize('frequent,rare',[(3,3),(10,3),(3,4)])
def test_supported_endpoint_is_not_lost_inside_broad_cohort(tmp_path,frequent,rare):
    config=config_for(tmp_path,view_prototypes_enabled=True)
    with FaceDatabase(config.database) as db:
        source=fixture(db,config,'1',[A]*3+[X]*frequent)
        appearance=fixture(db,config,'2',[C]*rare)
        rebuild_selected(db,[source,appearance],config)
        owner=candidates(db,source['scan'],config)[0][1]
        bank=catalogue(db.connection,config,'scope','ns',complete=True,
            usage=ReferenceUsage(config.data['clustering']['max_reference_bytes']))
        assert any(v.support_count>=3 and np.dot(v.group.samples[0],C)>.999 for v in bank)
        query=fixture(db,config,'3',[angle(84)]*3);rebuild_selected(db,[query],config)
        assert all(r[0]=='accepted' and r[1]==owner and r[2]['reference_role']=='attachment'
                   for r in candidates(db,query['scan'],config))


def test_single_endpoint_cannot_borrow_broad_cohort_support(tmp_path):
    from stash_ai.recognition.view_prototypes import confirmed_views
    config=config_for(tmp_path,view_prototypes_enabled=True)
    with FaceDatabase(config.database) as db:
        item=fixture(db,config,'1',[X]*4+[C])
        rows=[dict(r,person='owner') for r in db.connection.execute('SELECT * FROM observations WHERE scan=? ORDER BY cue',(item['scan'],))]
        bank=confirmed_views(db.connection,None,config.data['clustering'],rows=iter(rows),
            remaining_bytes=1048576,complete=True)
        assert bank and all(np.dot(v.group.samples[0],X)>.999 for v in bank)


@pytest.mark.parametrize('count',[2,3])
def test_best_supported_view_outvotes_weak_strict_mean_without_upgrading_authority(tmp_path,count):
    config=changed(config_for(tmp_path,view_prototypes_enabled=True),recognition={'min_identity_support':2})
    q=angle(50);rival=.8*q+.6*np.array([0.,0.,1.,0.])
    with FaceDatabase(config.database) as db:
        source=fixture(db,config,'1',[A]*3+[X]*3);rebuild_selected(db,[source],config)
        owner=candidates(db,source['scan'],config)[0][1]
        other=fixture(db,config,'2',[rival]*3);rebuild_selected(db,[other],config)
        assert candidates(db,other['scan'],config)[0][1]!=owner
        query=fixture(db,config,'3',[q]*count);rebuild_selected(db,[query],config)
        assert all(r[0]=='accepted' and r[1]==owner and r[2]['reference_role']=='attachment'
                   for r in candidates(db,query['scan'],config))


def test_coverage_centres_respect_explicit_reference_budget(tmp_path):
    from stash_ai.recognition.view_prototypes import confirmed_views
    config=config_for(tmp_path,view_prototypes_enabled=True)
    with FaceDatabase(config.database) as db:
        item=fixture(db,config,'1',[X]*3+[C]*3)
        rows=[dict(r,person='owner') for r in db.connection.execute('SELECT * FROM observations WHERE scan=? ORDER BY cue',(item['scan'],))]
        with pytest.raises(ValueError,match='exceed'):
            confirmed_views(db.connection,None,config.data['clustering'],rows=iter(rows),
                remaining_bytes=16,complete=True)
