"""Local matching retains the global bank's supported appearance coverage."""
import numpy as np
import pytest

from stash_ai.recognition.anchor_views import _confirmed
from stash_ai.recognition.clustering import matching_policy
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.global_views import catalogue
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.reference_usage import ReferenceUsage
from tests.test_recognition_anchor_views import config_for,X,C
from tests.test_recognition_frozen_partition import fixture


@pytest.mark.parametrize('count',[5,34])
@pytest.mark.parametrize('primary',[1,4])
def test_local_and_global_banks_keep_all_supported_reserve_cohorts(tmp_path,count,primary):
    config=config_for(tmp_path,max_view_prototypes_per_person=primary,context_view_enrichment=False)
    variants=[]
    for i in range(count):
        v=np.zeros(count+1);v[0]=np.sqrt(.7);v[i+1]=np.sqrt(.3);variants.append(v)
    with FaceDatabase(config.database) as db:
        item=fixture(db,config,'1',[v for v in variants for _ in range(3)])
        rebuild_selected(db,[item],config)
        local=_confirmed(db.connection,matching_policy(config),config.data['clustering'],ReferenceUsage(1048576))
        global_bank=catalogue(db.connection,config,'scope','ns',complete=True,usage=ReferenceUsage(1048576))
        assert len(local)==count
        assert {v.medoid for v in local}=={v.medoid for v in global_bank}
        assert all(v.support_count==3 for v in local)


@pytest.mark.parametrize('rare',[1,3])
def test_local_narrow_appearance_requires_its_own_votes(tmp_path,rare):
    config=config_for(tmp_path,max_view_prototypes_per_person=1,context_view_enrichment=False)
    with FaceDatabase(config.database) as db:
        item=fixture(db,config,'1',[X]*10+[C]*rare);rebuild_selected(db,[item],config)
        views=_confirmed(db.connection,matching_policy(config),config.data['clustering'],ReferenceUsage(1048576))
        assert any(np.dot(v.group.samples[0],C)>.999 for v in views)==(rare==3)
        assert any(np.dot(v.group.samples[0],X)>.999 for v in views)


def test_local_scratch_overflow_fails_instead_of_losing_late_looks(tmp_path):
    config=config_for(tmp_path,max_view_prototypes_per_person=1,context_view_enrichment=False)
    variants=[]
    for i in range(5):
        v=np.zeros(6);v[0]=np.sqrt(.7);v[i+1]=np.sqrt(.3);variants.append(v)
    with FaceDatabase(config.database) as db:
        item=fixture(db,config,'1',[v for v in variants for _ in range(3)]);rebuild_selected(db,[item],config)
        with pytest.raises(ValueError,match='Local anchor view evidence'):
            _confirmed(db.connection,matching_policy(config),config.data['clustering'],ReferenceUsage(64))
