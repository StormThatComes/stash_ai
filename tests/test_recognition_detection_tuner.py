"""Desktop event routing and single-owner detection, including a real Tk smoke test."""
from dataclasses import replace
import gc
import os
from pathlib import Path
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


def test_switching_bundled_pmpose_to_vitpose_and_back_keeps_panel_settings():
    pytest.importorskip('tkinter')
    from stash_ai.recognition.config import load
    from stash_ai.recognition.tune_editor import Settings

    config = load(Path(__file__).resolve().parents[1] / 'facial_recognition.yml')
    first, second = Settings(config), Settings(config)
    second.edit('pose.refinement.min_prompt_points', '3')
    second.edit('pose.estimator.backend', 'vitpose_plus_large')
    preview = second.effective().data['sapiensid']
    assert preview['pose']['backend'] == 'vitpose_plus_large'
    assert preview['pose']['refinement']['enabled'] and preview['pose']['deduplication']['enabled']
    assert 'pose.mask' in second.groups()
    assert 'pose.mask.apply_to' in second.groups()
    assert preview['collection']['min_sharpness'] > 0 and preview['identities']['anchor_min_sharpness'] > 0
    assert first.effective() is config

    second.edit('pose.estimator.backend', 'pmpose')
    pose = second.effective().data['sapiensid']['pose']
    assert pose['refinement']['enabled'] and pose['deduplication']['enabled']
    assert pose['refinement']['min_prompt_points'] == 3
    assert config.data['sapiensid']['pose']['refinement']['min_prompt_points'] == 1
    second.edit('pose.refinement.enabled', 'false')
    pose = second.effective().data['sapiensid']['pose']
    assert not pose['refinement']['enabled']
    assert pose['refinement']['masked_identity_crop'] is False


def test_tuner_vitpose_masks_keep_explicit_disable(tmp_path):
    from stash_ai.recognition.config import load
    from stash_ai.recognition.tune_editor import Settings
    path = tmp_path / 'vitpose.yml'
    path.write_text('version: 2\npose:\n  estimator: {backend: vitpose_plus_large}\n'
                    '  refinement: {enabled: true}\n')
    settings = Settings(load(path))
    assert settings.effective().data['sapiensid']['pose']['refinement']['enabled']
    settings.edit('pose.refinement.enabled', 'false')
    assert not settings.effective().data['sapiensid']['pose']['refinement']['enabled']


@pytest.fixture
def root():
    tk = pytest.importorskip('tkinter')
    try:
        window = tk.Tk()
    except tk.TclError as error:
        if os.getenv('STASH_AI_DESKTOP_TEST') == '1':
            pytest.fail(f'Desktop tests require a working display: {error}')
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
        ui.editor.tree.selection_set('face.detector.score_threshold')
        ui.editor.show_value()
        ui.editor.manual.set('0.83')
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


def test_phase_slider_reuses_detection_and_ab_positions_are_independent(root, tmp_path, monkeypatch):
    from stash_ai.recognition.detection_session import CueDetector
    from stash_ai.recognition.tune_window import Tuner
    from stash_ai.recognition import detection_phases as phases
    from stash_ai.recognition import detection_phase_timing as timing
    from tests.test_recognition_pose_refinement import BOX, pose_points, prediction
    case = setup(tmp_path, monkeypatch)
    case.services.assets = lambda *args, **kwargs: {}
    now = [0.]
    monkeypatch.setattr(timing, 'perf_counter', lambda: now[0])
    class Phased(Detector):
        def detect(self, image, **kwargs):
            now[0] += .125
            phases.pose(image, 'Before refinement', [BOX], prediction([pose_points()]))
            now[0] += 1.25
            phases.pose(image, 'After refinement', [BOX], prediction([pose_points()+1]))
            now[0] += .25
            return super().detect(image, **kwargs)
    detector = Phased()
    ui = Tuner(root, case.cfg, case.services,
               detector=CueDetector(case.services, factory=lambda *args: detector), compare_config=case.cfg)
    try:
        ui.scene_entry.insert(0, '1')
        ui.open_scene()
        settle(ui)
        assert ui.phases.slider.instate(['disabled'])
        ui.run_button.invoke()
        settle(ui)
        assert detector.calls == 1 and ui.cache['A'][1] is ui.cache['B'][1]
        assert ui.phases.positions == {'A': 3, 'B': 3}
        assert 'Step: 250.0 ms | Total: 1.625 s' in ui.images.panels['A'].caption.get()
        now[0] += 100  # Navigation must keep the original run's timings.
        ui.phases.slider.set(1)
        root.update()
        assert ui.phases.positions == {'A': 3, 'B': 1}
        assert 'Before refinement' in ui.images.panels['B'].caption.get()
        assert 'Step: 125.0 ms | Total: 125.0 ms' in ui.images.panels['B'].caption.get()
        assert 'Step: 125.0 ms | Total: 125.0 ms' in ui.phases.caption.get()
        ui.active.set('A')
        ui.choose_editor()
        ui.phases.step(-1)
        assert ui.phases.positions == {'A': 2, 'B': 1}
        assert 'After refinement' in ui.images.panels['A'].caption.get()
        assert 'Step: 1.250 s | Total: 1.375 s' in ui.images.panels['A'].caption.get()
        ui.phases.final.invoke()
        assert ui.phases.positions == {'A': 3, 'B': 1}
        ui.active.set('B')
        ui.choose_editor()
        ui.edited()
        assert ui.phases.slider.instate(['disabled']) and list(ui.phases.results) == ['A']
        ui.step(1)
        settle(ui)
        assert not ui.phases.results and not ui.phases.positions
        assert detector.calls == 1 and not case.stash.writes
    finally:
        ui.close()


def test_hover_tracks_phase_zoom_scroll_and_clears_without_model_calls(root, tmp_path, monkeypatch):
    from stash_ai.recognition.detection_session import CueDetector
    from stash_ai.recognition.tune_window import Tuner
    from stash_ai.recognition import detection_phases as phases
    from tests.test_recognition_pose_refinement import BOX, pose_points, prediction
    case = setup(tmp_path, monkeypatch)
    case.services.assets = lambda *args, **kwargs: {}
    case.cfg.data['sapiensid']['pose']['backend'] = 'pmpose'
    class Phased(Detector):
        def detect(self, image, **kwargs):
            value = prediction([pose_points()])
            phases.pose(image, 'First pass', [BOX], value)
            value[1][:] = .72
            phases.pose(image, 'Second pass', [BOX], value)
            return super().detect(image, **kwargs)
    detector = Phased()
    ui = Tuner(root, case.cfg, case.services, detector=CueDetector(case.services, factory=lambda *args: detector),
               compare_config=case.cfg)
    try:
        ui.scene_entry.insert(0, '1')
        ui.open_scene()
        settle(ui)
        ui.run_button.invoke()
        settle(ui)
        ui.phases.slide(1)  # B: first pass; A stays final.
        root.update()
        panel = ui.images.panels['B']
        def hover():
            x = round(35*panel.zoom-panel.canvas.canvasx(0))
            y = round(20*panel.zoom-panel.canvas.canvasy(0))
            panel.canvas.event_generate('<Motion>', x=x, y=y)
            root.update()
        hover()
        assert 'nose (#0)' in panel.hover.text and '0.9500' in panel.hover.text
        assert panel.hover.window.winfo_viewable()
        assert not ui.images.panels['A'].hover.text
        ui.phases.slide(2)
        assert not panel.hover.text
        root.update()
        hover()
        assert 'Score (expected OKS): 0.7200' in panel.hover.text
        ui.zoom.set('400%')
        ui.set_zoom()
        root.update()
        ui.images.pan('x', 'moveto', .02)
        root.update()
        hover()
        assert 'Score (expected OKS): 0.7200' in panel.hover.text
        panel.canvas.event_generate('<Leave>')
        root.update()
        assert not panel.hover.text and not panel.hover.window.winfo_viewable()
        hover()
        ui.edited()
        assert not panel.hover.points and not panel.hover.text
        ui.step(1)
        settle(ui)
        assert all(not p.hover.points for p in ui.images.panels.values())
        assert detector.calls == 1
    finally:
        ui.close()


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
            ui.editor.tree.selection_set('face.detector.score_threshold')
            ui.editor.show_value()
            ui.editor.manual.set(str(score))
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
        react.write_text('version: 2\nface: {detector: {nms_threshold: 1}}\n'
                         'pose: {estimator: {backend: vitpose_plus_large}, refinement: {enabled: true}}\n')
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


def test_parameter_sidebar_auto_reset_and_diff(root):
    import yaml
    from stash_ai.recognition.config import load
    from stash_ai.recognition.tune_editor import ParameterEditor, Settings

    settings = Settings(load(Path(__file__).resolve().parents[1] / 'facial_recognition.yml'))
    changes = []
    editor = ParameterEditor(root, settings, changes.append, pytest.fail)
    editor.pack()
    editor.query.set('min_recovery_mask_score')
    root.update()
    path = 'pose.mask.min_recovery_mask_score'
    assert editor.path == path
    editor.commit('0.71')
    editor.commit('0.71')
    assert changes == [path]
    editor.copy_diff()
    assert yaml.safe_load(root.clipboard_get()) == {'pose': {'mask': {'min_recovery_mask_score': .71}}}
    editor.auto.invoke()
    assert settings.value(path) is None
    editor.undo()
    assert settings.value(path) == .71
    editor.reset()
    assert settings.value(path) is None


def test_live_masks_edit_only_b_and_discard_late_preview(root):
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.tune_editor import Settings
    from stash_ai.recognition.tune_mask_candidate import MaskSample
    from stash_ai.recognition.tune_mask_view import LiveMasks

    panel = LiveMasks(root, pytest.fail)
    panel.pack(fill='both', expand=True)
    config = load(Path(__file__).resolve().parents[1] / 'facial_recognition.yml')
    a, b = Settings(config), Settings(config)
    raw = np.zeros((40, 80), bool)
    raw[8:26, 12:31] = True
    raw[31:34, 35:39] = True
    own = np.array([[20., 15.]])
    sample = MaskSample('P0', raw.shape, np.packbits(raw).tobytes(), .98, (0, 0, 80, 40),
                        own, own, np.empty((0, 2)), np.empty((0, 2, 2)))
    history = SimpleNamespace(samples=(sample,))
    def finish():
        deadline = time.monotonic() + 5
        while any(w.future is not None for w in panel.workers.values()):
            assert time.monotonic() < deadline
            panel.poll()
            root.update()
            time.sleep(.002)
    try:
        with Image.new('RGB', (80, 40)) as image:
            panel.configure_panel('B', True)
            panel.activate(True)
            for name, cfg in (('A', a), ('B', b)):
                cfg.edit('pose.mask.max_island_gap_fraction', '0')
                panel.set_samples(name, history, cfg, image)
            finish()
            for size in (14, 8, 13):
                b.edit('pose.mask.min_island_pixels', str(size))
                panel.request('B')
            finish()
            assert panel.details['A'][1][32, 36]
            assert not panel.details['B'][1][32, 36]
            panel.request('B')
            panel.clear('B')
            finish()
            assert 'B' not in panel.details
    finally:
        panel.close()
