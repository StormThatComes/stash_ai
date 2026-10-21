"""New owner appearances preserve proofs; rare confirmed cohorts remain usable."""
import json
import numpy as np
import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.historical_references import capture
from stash_ai.recognition.historical_references import context
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_anchor_views import config_for,A,X,C
from tests.test_recognition_frozen_partition import fixture
from tests.test_recognition_scene_context import candidates


@pytest.mark.parametrize('legacy',[False,True])
def test_adding_owner_appearance_preserves_original_auxiliary_confirmation(tmp_path,legacy):
    config=config_for(tmp_path,view_prototypes_enabled=True)
    with FaceDatabase(config.database) as db:
        first=fixture(db,config,'1',[A]*3+[X]*3,range(3,6));rebuild_selected(db,[first],config)
        original=candidates(db,first['scan'],config)
        assert all(r[2]['strict_confirmation']['support_ids'] for r in original[3:])
        if legacy:
            db.connection.execute("UPDATE candidates SET payload=json_remove(payload,'$.strict_confirmation') WHERE scan=?",
                (first['scan'],));db.connection.commit()
        before=capture(db.connection,[],config,'scope','ns','before')
        opposite=np.array([.7,-np.sqrt(1-.7**2),0.,0.])
        second=fixture(db,config,'2',[opposite]*3);rebuild_selected(db,[second],config)
        after=capture(db.connection,[],config,'scope','ns','after')
        assert sum(v.authority=='verified_view' for v in before.views)==1
        assert sum(v.authority=='verified_view' for v in after.views)==1
        query=fixture(db,config,'3',[C]*3);rebuild_selected(db,[query],config)
        assert all(r[0]=='accepted' and r[1]==original[0][1] for r in candidates(db,query['scan'],config))
        # Removing an original witness revokes the persisted proof.
        if not legacy:
            witness=original[3][2]['strict_confirmation']['support_ids'][0]
            db.connection.execute('DELETE FROM members WHERE observation=?',(witness,));db.connection.commit()
            invalid=capture(db.connection,[],config,'scope','ns','removed')
            assert not any(v.authority=='verified_view' for v in invalid.views)


@pytest.mark.parametrize('joint',[False,True])
@pytest.mark.parametrize('appearances',[5,34])
def test_rare_confirmed_cohort_survives_primary_and_discovery_limits(tmp_path,joint,appearances):
    config=config_for(tmp_path,max_view_prototypes_per_person=4)
    variants=[]
    for i in range(1,appearances+1):
        v=np.zeros(appearances+1);v[0]=np.sqrt(.7);v[i]=np.sqrt(.3);variants.append(v)
    query=np.zeros(appearances+1);query[0]=.35/np.sqrt(.7);query[-1]=np.sqrt(1-query[0]**2)
    with FaceDatabase(config.database) as db:
        source=fixture(db,config,'1',[v for i,v in enumerate(variants) for _ in range(3 if i==appearances-1 else 4)])
        if not joint:rebuild_selected(db,[source],config)
        second=fixture(db,config,'2',[query]*3)
        rebuild_selected(db,[source,second] if joint else [second],config)
        rows=[r for item in (source,second) for r in candidates(db,item['scan'],config)]
        assert all(r[0]=='accepted' for r in rows)
        assert len({r[1] for r in rows})==1


def test_changed_confirmation_proof_invalidates_external_context(tmp_path):
    config=config_for(tmp_path,view_prototypes_enabled=True)
    with FaceDatabase(config.database) as db:
        first=fixture(db,config,'1',[A]*3+[X]*3,range(3,6));rebuild_selected(db,[first],config)
        before=context(db.connection,[],config,'scope','ns')
        db.connection.execute("UPDATE candidates SET payload=json_set(payload,'$.strict_confirmation.support_ids',json('[\"missing\"]')) "
            "WHERE scan=? AND json_extract(payload,'$.reason')='auxiliary_identity'",(first['scan'],))
        db.connection.commit()
        assert context(db.connection,[],config,'scope','ns')!=before
        assert not any(v.authority=='verified_view' for v in capture(db.connection,[],config,'scope','ns','changed').views)
