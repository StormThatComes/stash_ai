"""Desktop event routing and single-owner detection, including a real Tk smoke test."""
from dataclasses import replace
import gc
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
    gc.collect()


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
        assert 'Matched faces' in ui.images.panels['A'].caption.get()
        ui.step(1)
        settle(ui)
        assert ui.images.panels['A'].caption.get() == 'Original VTT cue'
        assert [item.calls for item in loaded] == [1]
    finally:
        ui.close()
    assert all(item.closed for item in loaded)
    assert not case.stash.writes


def test_comparison_reuses_a_retries_b_and_releases_models_on_owner(root, tmp_path, monkeypatch):
    from stash_ai.recognition.detection_session import CueDetector
    from stash_ai.recognition.tune_window import Tuner
    case = setup(tmp_path, monkeypatch)
    case.services.assets = lambda *args, **kwargs: {}
    events = []

    class Owned(Detector):
        def __init__(self, score):
            super().__init__()
            self.score = score
            events.append(('open', score, threading.get_ident()))

        def detect(self, image, **kwargs):
            events.append(('run', self.score, threading.get_ident()))
            if self.score == .77:
                raise RuntimeError('Simulated model failure.')
            return super().detect(image, **kwargs)

        def close(self):
            events.append(('close', self.score, threading.get_ident()))
            super().close()

    def factory(config, paths):
        return Owned(config.data['detection']['score_threshold'])

    ui = Tuner(root, case.cfg, case.services, detector=CueDetector(case.services, factory=factory),
               compare_config=replace(case.cfg, source_path=tmp_path / 'same-settings.yml'))
    try:
        ui.scene_entry.insert(0, '1')
        ui.open_scene()
        settle(ui)
        ui.run_button.invoke()
        settle(ui)
        assert ui.cache['A'][1] is ui.cache['B'][1]
        a = ui.cache['A']
        for score in (.77, .82):
            ui.editor.parameter.set('score_threshold')
            ui.editor.show_value()
            ui.editor.value_input.set(str(score))
            ui.editor.apply()
            assert ui.cache == {'A': a}
            ui.run_button.invoke()
            settle(ui)
            caption = ui.images.panels['B'].caption.get()
            expected = 'Simulated model failure' if score == .77 else 'Matched faces'
            assert expected in caption
            assert ui.cache['A'] is a
        ui.run_button.invoke()
        settle(ui)
        assert [score for action, score, _ in events if action == 'run'] == [.9, .77, .82]
        ui.copy_a()
        ui.run_button.invoke()
        settle(ui)
        assert ui.cache['B'][1] is a[1]
        ui.zoom.set('400%')
        ui.set_zoom()
        root.update()
        ui.images.pan('x', 'moveto', .4)
        root.update()
        panels = ui.images.panels
        assert abs(panels['A'].canvas.canvasx(0) - panels['B'].canvas.canvasx(0)) <= 1
        assert all(panel.photo is not None for panel in panels.values())
        react = tmp_path / 'different.yml'
        react.write_text('version: 2\nface:\n  detector:\n    nms_threshold: 1\n')
        ui.path.delete(0, 'end')
        ui.path.insert(0, str(react))
        ui.load_yaml()
        assert ui.settings['B'].effective().data['detection']['nms_threshold'] == 1
        ui.settings['B'].edit('face.detector.nms_threshold', '.61')
        assert ui.settings['B'].effective().data['detection']['nms_threshold'] == .61
        assert ui.cache == {'A': a}
    finally:
        ui.close()
    assert [event for event, _, _ in events] == ['open', 'run', 'close'] * 3
    assert len({owner for _, _, owner in events}) == 1
    assert events[0][2] != threading.get_ident()


def test_navigating_during_inference_discards_old_result_and_pending_b(root, tmp_path, monkeypatch):
    from stash_ai.recognition.tune_window import Tuner
    case = setup(tmp_path, monkeypatch)
    started, release = threading.Event(), threading.Event()
    measured = []

    class Slow:
        def render(self, frame, config):
            measured.append(frame.index)
            started.set()
            assert release.wait(5)
            return frame.image.copy(), {'faces': [], 'people': []}

        def close(self):
            pass

    ui = Tuner(root, case.cfg, case.services, detector=Slow(), compare_config=case.cfg)
    try:
        ui.scene_entry.insert(0, '1')
        ui.open_scene()
        settle(ui)
        ui.settings['B'].edit('face.detector.score_threshold', '.82')
        ui.edited()
        ui.run_button.invoke()
        assert started.wait(2)
        ui.step(1)
        deadline = time.monotonic() + 3
        while ui.frame is None and time.monotonic() < deadline:
            root.update()
            time.sleep(.005)
        assert ui.frame.index == 1
        release.set()
        settle(ui)
        assert measured == [0] and not ui.cache
        assert all(panel.caption.get() == 'Original VTT cue' for panel in ui.images.panels.values())
    finally:
        release.set()
        ui.close()
