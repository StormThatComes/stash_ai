import pytest
from stash_ai.recognition.cli import parser
from stash_ai.recognition.merge_goals import resolve
from stash_ai.recognition.fragmentation_report import _open_readonly
from tests.test_recognition_merge_report import seeded


def test_sets_and_alias_overlap(tmp_path):
    config, people = seeded(tmp_path)
    with _open_readonly(config.database) as db:
        result = resolve(db, config, [['900', '900'], ['901']])
        assert len(result['groups'][0]['members']) == 1
        assert len(result['groups'][0]['members'][0]['observations']) == 3
        with pytest.raises(ValueError, match='Overlapping'):
            resolve(db, config, [['900'], ['900']])
    args = parser().parse_args(['report-merge', '--performer-set', '900,901', '--performer-set', '902'])
    assert args.performer_set == [['900', '901'], ['902']]
    assert len(parser().parse_args(['report-merge', '--identity-id', 'a', '--identity-id', 'b', '--identity-id', 'c']).identity_id) == 3
