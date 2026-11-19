"""Phase timings measure detection once, independently of copying and browsing."""
from types import SimpleNamespace

from PIL import Image
import pytest

from stash_ai.recognition import detection_phases as phases
from stash_ai.recognition import detection_phase_timing as timing


@pytest.fixture
def clock(monkeypatch):
    class ManualClock:
        now = 100.

        def advance(self, seconds):
            self.now += seconds
    clock = ManualClock()
    monkeypatch.setattr(timing, 'perf_counter', lambda: clock.now)
    return clock


def test_phase_intervals_exclude_snapshot_work_and_freeze_before_preview(clock, monkeypatch):
    from tests.test_recognition_pose_refinement import BOX, pose_points, prediction
    save, xy = phases.History._save, phases._xy
    def slow_save(self, *args):
        clock.advance(10)
        save(self, *args)
        clock.advance(20)
    def slow_copy(points):
        clock.advance(5)
        return xy(points)
    monkeypatch.setattr(phases.History, '_save', slow_save)
    monkeypatch.setattr(phases, '_xy', slow_copy)
    with Image.new('RGB', (160, 100)) as image, phases.capture(image) as history:
        clock.advance(.125)
        phases.boxes(image, [BOX])
        clock.advance(1.25)
        phases.pose(image, 'Pose pass', [BOX], prediction([pose_points()]))
        clock.advance(.25)
        phases.finish(image)
        clock.advance(500)  # Annotation and preview work must not leak into totals.
        phases.finish(image)
    assert [p.timing for p in history.phases] == [timing.Timing(0, 0), timing.Timing(.125, .125),
                                               timing.Timing(1.25, 1.375)]
    assert history.clock.final == timing.Timing(.25, 1.625)
    assert sum(p.timing.seconds for p in history.phases) + history.clock.final.seconds == 1.625
    assert timing.caption(history, 1) == ' | Step: 125.0 ms | Total: 125.0 ms'
    assert timing.caption(history, 3) == ' | Step: 250.0 ms | Total: 1.625 s'


def test_omitted_phases_remain_in_final_total(clock, monkeypatch):
    monkeypatch.setattr(phases, 'MAX_PHASES', 2)
    with Image.new('RGB', (20, 20)) as image, phases.capture(image) as history:
        clock.advance(1)
        phases.boxes(image, [])
        clock.advance(2)
        phases.notes(image, 'Omitted phase', [])
        clock.advance(3)
    assert len(history.phases) == 2 and history.omitted == 1
    assert history.clock.final == timing.Timing(5, 6)
    assert 'Remaining (includes omitted phases): 5.000 s | Total: 6.000 s' in timing.caption(history, 2)


def test_disabled_capture_never_reads_clock(monkeypatch):
    monkeypatch.setattr(timing, 'perf_counter', lambda: pytest.fail('timed ordinary recognition'))
    with Image.new('RGB', (20, 20)) as image, phases.capture(image, enabled=False):
        phases.boxes(image, [])
        phases.pose(image, 'Unused', [], ())
        phases.finish(image)


@pytest.mark.parametrize('empty', [False, True])
def test_cue_render_excludes_model_loading_and_annotation_even_without_people(clock, monkeypatch, tmp_path, empty):
    from stash_ai.recognition import detection_audit as audit
    from stash_ai.recognition.detection_session import CueDetector, CueFrame
    from tests.test_recognition_detection_audit import Detector, setup
    case = setup(tmp_path, monkeypatch)
    class Timed(Detector):
        def detect(self, image, **kwargs):
            clock.advance(.5)
            if not empty:
                phases.boxes(image, [[4, 4, 156, 156]])
            clock.advance(.25)
            return super().detect(image, **kwargs)
    detector = Timed()
    detector.calls = 2 if empty else 0
    def factory(*args):
        clock.advance(100)
        return detector
    layout = audit.layout_faces
    def slow_layout(*args):
        clock.advance(200)
        return layout(*args)
    monkeypatch.setattr(audit, 'layout_faces', slow_layout)
    engine = CueDetector(case.services, factory=factory)
    try:
        with Image.new('RGB', (160, 160)) as image:
            cue = SimpleNamespace(start=0., end=1., rect=(0, 0, 160, 160))
            result = engine.render_phases(CueFrame('1', 0, cue, image), case.cfg)
            assert result.phases.clock.final == timing.Timing(.75 if empty else .25, .75)
            assert len(result.phases.phases) == (1 if empty else 2)
            assert result.phases.image is None
            assert 'timing' not in result[1]  # Export/cache schemas are unchanged.
            result[0].close()
    finally:
        engine.close()
