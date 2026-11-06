from pathlib import Path
import json
import pytest
from stash_ai.recognition import rebuild
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_replay_checkpoints import configured,selection,interrupt
from tests.test_recognition_clustering import setup


@pytest.mark.parametrize('storage',['disk','memory'])
def test_resume_skips_raw_import_and_canonicalization(tmp_path,monkeypatch,storage):
    config=configured(setup(tmp_path),rebuild_storage=storage)
    with FaceDatabase(config.database) as state:
        selected=selection(state,config)
        interrupt(monkeypatch,state,selected,config,after=2)
        def unexpected(*args,**kwargs):raise AssertionError('raw replay was copied again')
        monkeypatch.setattr(rebuild,'_copy_selection',unexpected)
        rebuild.rebuild_selected(state,selected,config)
        assert state.last_rebuild_profile['counts']['input_snapshot_restores']==1


def test_corrupt_input_image_falls_back_before_grouping_resume(tmp_path,monkeypatch):
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config)
        interrupt(monkeypatch,state,selected,config,after=2)
        directory=Path(str(config.database)+'.rebuild-input')
        manifest=json.loads((directory/'current.json').read_text())
        (directory/manifest['file']).write_bytes(b'corrupt')
        rebuild.rebuild_selected(state,selected,config)
        assert state.last_rebuild_profile['counts']['input_cache_rejections']==1
        assert state.last_rebuild_profile['counts']['checkpoint_resumed_scenes']==2
