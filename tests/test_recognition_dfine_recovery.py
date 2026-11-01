"""Reject bad proposals without losing healthy detections or hiding model failures."""
from contextlib import ExitStack
import logging
from types import SimpleNamespace

import pytest
from PIL import Image

from stash_ai.recognition import dfine_detector as dfine


def output(frames=1, queries=3):
    import torch
    logits = torch.full((frames, queries, 80), -20.)
    logits[..., 0] = 8.
    boxes = torch.tensor([.5, .5, .8, .8]).repeat(frames, queries, 1)
    return SimpleNamespace(logits=logits, pred_boxes=boxes)


@pytest.mark.parametrize('axis,value', [(2, -.1), (3, -.1), (2, 0.), (3, 0.)])
@pytest.mark.parametrize('score', [-20., 20.])
@pytest.mark.parametrize('suppress', [False, True])
def test_invalid_geometry_cannot_abort_or_consume_valid_result_limit(axis, value, score, suppress, caplog):
    import torch
    caplog.set_level(logging.DEBUG, logger=dfine.__name__)
    prediction = output()
    prediction.pred_boxes[0, 0, axis] = value
    prediction.logits[0, 0, 0] = score
    prediction.logits[0, 2, 0] = 6.
    original = prediction.pred_boxes.clone()
    result = dfine.DFineDetector._decode(prediction, [(100, 200)], .5, .7, 1, suppress)[0].boxes
    torch.testing.assert_close(result.xyxy, torch.tensor([[10., 20., 90., 180.]]))
    torch.testing.assert_close(result.conf, torch.tensor([8.]).sigmoid())
    torch.testing.assert_close(prediction.pred_boxes, original)
    assert 'discarded 1 proposal(s)' in caplog.text
    assert 'input_frame=0, query=0' in caplog.text


@pytest.mark.parametrize('log_level', [logging.WARNING, logging.DEBUG])
def test_all_invalid_frame_keeps_other_frame_and_empty_result_shape(caplog, log_level):
    import torch
    caplog.set_level(log_level, logger=dfine.__name__)
    prediction = output(frames=2)
    prediction.pred_boxes[0, :, 2] = -1.
    result = dfine.DFineDetector._decode(prediction, [(100, 200), (50, 100)], .5, .7, 10, True)
    assert result[0].boxes.xyxy.shape == (0, 4)
    assert result[0].boxes.conf.shape == (0,)
    torch.testing.assert_close(result[1].boxes.xyxy, torch.tensor([[5., 10., 45., 90.]]))
    if log_level == logging.DEBUG:
        assert len(caplog.records) == 1 and 'discarded 3' in caplog.text
        assert caplog.records[0].levelno == logging.DEBUG
    else:
        assert not caplog.records


@pytest.mark.parametrize('tensor,component', [('logits', 0), ('logits', 79), ('pred_boxes', 2)])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_diagnostics_identify_tensor_frame_query_and_component(tensor, component, value):
    prediction = output(frames=2)
    getattr(prediction, tensor)[1, 2, component] = value
    with pytest.raises(dfine._NonFiniteDetections) as caught:
        dfine.DFineDetector._decode(prediction, [(100, 200)] * 2, .5, .7, 10, True, frame_offset=4)
    assert caught.value.frames == {1}
    message = str(caught.value)
    assert f'{tensor}: count=1, input_frame=5, query=2' in message
    assert f'={component}, value={value}' in message
    assert 'NaN/Inf' in message


class Processor:
    def __call__(self, *, images, return_tensors):
        import torch
        from transformers import BatchFeature
        return BatchFeature({'pixel_values': torch.tensor(
            [image.getpixel((0, 0))[0] for image in images], dtype=torch.float32).reshape(-1, 1, 1, 1)})


def predict(model, count=4, batch_size=2):
    detector = dfine.DFineDetector(model, Processor(), device='cpu', batch_size=batch_size)
    with ExitStack() as stack:
        images = [stack.enter_context(Image.new('RGB', (100, 200), (index, 0, 0))) for index in range(count)]
        return detector.predict(images, conf=.5, iou=.7, max_det=10)


def test_retry_only_affected_frames_preserves_healthy_predictions_and_order(caplog):
    import torch
    calls = []
    def model(pixel_values):
        markers = pixel_values.flatten().int().tolist()
        calls.append(markers)
        prediction = output(frames=len(markers), queries=1)
        # Healthy batch results must remain untouched, not be recomputed.
        prediction.logits[..., 0] = len(calls)
        for row, marker in enumerate(markers):
            prediction.pred_boxes[row, 0, 0] = .45 + marker * .01
            if marker in (1, 3) and len(markers) > 1:
                prediction.pred_boxes[row, 0, 2] = float('nan')
        return prediction
    result = predict(model)
    assert calls == [[0, 1], [1], [2, 3], [3]]
    assert len(result) == 4
    for index, item in enumerate(result):
        torch.testing.assert_close(item.boxes.conf, torch.tensor([float(index + 1)]).sigmoid())
        torch.testing.assert_close(item.boxes.xyxy[:, 0], torch.tensor([5. + index]))
    assert 'input_frame=3' in caplog.text
    assert 'Retrying 1 affected frame(s) once individually in FP32 on cpu' in caplog.text


def test_multiple_affected_frames_are_retried_once_each():
    calls = []
    def model(pixel_values):
        markers = pixel_values.flatten().int().tolist()
        calls.append(markers)
        prediction = output(frames=len(markers), queries=1)
        if len(markers) > 1:
            prediction.logits[[0, 2], 0, 0] = float('inf')
        return prediction
    assert len(predict(model, count=3, batch_size=3)) == 3
    assert calls == [[0, 1, 2], [0], [2]]


@pytest.mark.parametrize('count', [1, 2])
def test_persistent_nonfinite_failure_is_not_saved_as_empty_detections(count):
    calls = []
    def model(pixel_values):
        markers = pixel_values.flatten().int().tolist()
        calls.append(markers)
        prediction = output(frames=len(markers))
        prediction.pred_boxes[-1, 1, 3] = float('nan')
        return prediction
    with pytest.raises(ValueError, match='Persistent after one isolated FP32 retry on cpu') as caught:
        predict(model, count=count)
    assert len(calls) == 2 and calls[-1] == [count-1]
    assert f'input_frame={count-1}, query=1, cx/cy/w/h=3' in str(caught.value)


@pytest.mark.parametrize('failure', ['shape', 'runtime'])
def test_non_numeric_failures_are_not_retried(failure):
    calls = []
    def model(pixel_values):
        calls.append(len(pixel_values))
        if failure == 'runtime':
            raise RuntimeError('inference failure')
        prediction = output(frames=len(pixel_values))
        prediction.logits = prediction.logits[..., :79]
        return prediction
    error, message = (ValueError, 'tensor shapes') if failure == 'shape' else (RuntimeError, 'inference failure')
    with pytest.raises(error, match=message):
        predict(model)
    assert calls == [2]


def test_dfine_disables_outer_autocast_and_restores_caller_context():
    import torch
    linear = torch.nn.Linear(1, 80).float().eval()
    def model(pixel_values):
        logits = linear(pixel_values.reshape(-1, 1, 1))
        assert logits.dtype == torch.float32
        assert not torch.is_autocast_enabled('cpu')
        assert not torch.is_grad_enabled()
        prediction = output(frames=len(pixel_values), queries=1)
        prediction.logits = logits
        return prediction
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        predict(model)
        assert torch.is_autocast_enabled('cpu')


@pytest.mark.parametrize('verbose', [False, True])
def test_geometry_warning_uses_console_report_when_available(monkeypatch, caplog, verbose):
    from stash_ai.runtime import console
    from stash_ai.runtime.report import RunReport
    caplog.set_level(logging.DEBUG, logger=dfine.__name__)
    messages = []
    report = RunReport(verbose=verbose, write=messages.append)
    monkeypatch.setattr(console, 'current_report', lambda: report)
    prediction = output()
    prediction.pred_boxes[0, 0, 3] = -1.
    dfine.DFineDetector._decode(prediction, [(100, 200)], .5, .7, 10, True)
    if verbose:
        assert len(messages) == 1 and messages[0].startswith('Warning: D-FINE discarded')
    else:
        assert not messages
    assert not caplog.records
