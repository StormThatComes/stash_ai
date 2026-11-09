"""Desktop event routing and single-owner detection, including a real Tk smoke test."""
import threading
import time

import pytest

from stash_ai.recognition.tune_worker import Worker
from tests.test_recognition_detection_audit import Detector, setup


def test_busy_worker_coalesces_requests_and_closes_on_owner_thread():
    worker = Worker('test-owner')
    started, release = threading.Event(), threading.Event()
    calls, displayed = [], []

    def operation(key):
        calls.append((key, threading.get_ident()))
        if key == 0:
            started.set()
            assert release.wait(3)
        return key

    try:
        worker.request(lambda: operation(0), displayed.append, pytest.fail)
        assert started.wait(3)
        for key in range(1, 30):
            worker.request(lambda k=key: operation(k), displayed.append, pytest.fail)
        release.set()
        while worker.future is not None:
            worker.future.result(timeout=3)
            worker.poll()
    finally:
        release.set()
        worker.close(lambda: operation('close'))
    assert [key for key, _ in calls] == [0, 29, 'close']
    assert len({owner for _, owner in calls}) == 1
    assert displayed == [29]


@pytest.fixture
def root():
    tk = pytest.importorskip('tkinter')
    try:
        window = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f'No graphical desktop: {error}')
    yield window
    window.destroy()


def settle(ui):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        ui.root.update()
        if ui.media.future is None and ui.inference.future is None:
            return
        time.sleep(.005)
    pytest.fail('Desktop workers did not finish.')


def test_only_run_button_loads_detectors_and_navigation_clears_annotations(root, tmp_path, monkeypatch):
    from stash_ai.recognition.detection_session import CueDetector
    from stash_ai.recognition.tune_window import Tuner
    case = setup(tmp_path, monkeypatch)
    case.services.assets = lambda *args, **kwargs: {}
    loaded = []
    def factory(cfg, paths):
        loaded.append(Detector())
        return loaded[-1]

    ui = Tuner(root, case.cfg, case.services, detector=CueDetector(case.services, factory=factory))
    try:
        ui.scene_entry.insert(0, '1')
        ui.open_scene()
        settle(ui)
        for index in (1, 0, 2, 1, 2):
            ui.slider.set(index)
        ui.step(-1)
        settle(ui)
        ui.editor.parameter.set('score_threshold')
        ui.editor.show_value()
        ui.editor.value_input.set('0.83')
        ui.editor.apply()
        assert not loaded and ui.frame.index == 1
        ui.run_button.invoke()
        settle(ui)
        assert [item.calls for item in loaded] == [1]
        assert 'Matched faces' in ui.panel.caption.get()
        ui.step(1)
        settle(ui)
        assert ui.panel.caption.get() == 'Original VTT cue'
        assert [item.calls for item in loaded] == [1]
    finally:
        ui.close()
    assert all(item.closed for item in loaded)
    assert not case.stash.writes
