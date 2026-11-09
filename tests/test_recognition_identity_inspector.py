import gc
import time

import pytest

from stash_ai.recognition.cli import parser
from stash_ai.recognition.identity_inspection import IdentityReader
from tests.test_recognition_identity_report import seed


def test_paged_inspection_reads_only_selected_crop(tmp_path, monkeypatch):
    from stash_ai.recognition import identity_inspection
    config, person, _ = seed(tmp_path, 6)
    before = config.database.read_bytes()
    monkeypatch.setattr(identity_inspection, 'PAGE_SIZE', 2)
    reader = IdentityReader(config, performer_id='900')
    with monkeypatch.context() as context:
        context.setattr(identity_inspection, 'cold_crop', lambda *_: pytest.fail('List loaded pixels'))
        rows, more = reader.page(scene='2', cue=0)
        assert len(rows) == 2 and not more
        first, more = reader.page()
        assert more
        second, _ = reader.page(offset=2)
        assert {row['id'] for row in first}.isdisjoint(row['id'] for row in second)
    address = rows[0]['id']
    filtered, _ = reader.page(observation=address)
    assert filtered == rows[:1]
    detail = reader.detail('face', address)
    assert detail['image'][0].size == (37,43)
    assert detail['trace']['decisions'][0]['metrics']['score'] == .87
    assert reader.reference('face', 'missing')[0] is None
    assert config.database.read_bytes() == before
    args = parser().parse_args(['inspect-identity','--identity-id',person,'--observation-id',address])
    assert args.observation_id == address


def test_native_window_selects_observation_and_displays_recorded_reason(tmp_path):
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except tk.TclError as error:
        import os
        if os.environ.get('STASH_AI_DESKTOP_TEST'):
            raise
        pytest.skip(f'No graphical desktop: {error}')
    from stash_ai.recognition.identity_window import Inspector
    config, person, _ = seed(tmp_path, 1)
    reader = IdentityReader(config, identity_id=person)
    address = reader.page()[0][0]['id']
    ui = Inspector(root, reader, 'Test identity', address)
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            root.update()
            if address in ui.status.get() and 'known_identity' in ui.notes.get('1.0','end'):
                break
            time.sleep(.005)
        assert 'known_identity' in ui.notes.get('1.0','end')
        assert ui.images.panels['A'].image.size == (37,43)
        assert 'No recorded reference' in ui.images.panels['B'].caption.get()
    finally:
        ui.close()
        root.destroy()
        del ui, root
        gc.collect()
