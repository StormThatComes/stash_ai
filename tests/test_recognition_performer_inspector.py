from stash_ai.recognition.cli import parser
from stash_ai.recognition.identity_inspection import IdentityReader
from stash_ai.recognition.performer_inspection import summary
from tests.test_recognition_identity_inspector import scene_fixture


def test_performer_summary_matches_current_accepted_pages(tmp_path):
    config, person = scene_fixture(tmp_path)
    before = config.database.read_bytes()
    reader = IdentityReader(config, performer_id='900')
    result = summary(reader)
    rows, _ = reader.page(accepted_only=True)
    assert result['observations'] == len(rows) == 7
    assert result['scenes'] == 2
    assert result['person']['id'] == person
    assert config.database.read_bytes() == before
    assert parser().parse_args(['inspect-performer','--performer-id','900','--offline']).offline
