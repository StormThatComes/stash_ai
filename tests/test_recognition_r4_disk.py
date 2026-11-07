from pathlib import Path
from types import SimpleNamespace
from stash_ai.recognition.disk_budget import scope,reserve_file


def test_file_reservations_share_one_volume_and_release_after_failure(tmp_path):
    path=tmp_path/'db'
    config=SimpleNamespace(database=path,data={'execution':{'disk_total_bytes':1000,'disk_free_bytes':0}})
    path.write_bytes(b'x'*100)
    a,b=(Path(str(path)+suffix) for suffix in ('.a','.b'))
    a.touch();b.touch()
    with scope(config):
        with reserve_file(a,700) as first:
            assert first==700
            a.write_bytes(b'x'*300)
            with reserve_file(b,500) as second:assert second==200
        with reserve_file(b,900) as freed:assert freed==600
