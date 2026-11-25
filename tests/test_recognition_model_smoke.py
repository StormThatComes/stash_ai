"""Explicit CPU loading/alignment smoke, not a face accuracy benchmark."""
import os
import pytest


@pytest.mark.skipif(os.getenv('STASH_AI_RECOGNITION_SMOKE') != '1', reason='explicit pinned face-model smoke')
@pytest.mark.parametrize('recognizer,dimensions', [('sface', 128), ('auraface', 512)])
def test_pinned_face_models_on_cpu(tmp_path, recognizer, dimensions):
    import numpy as np
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.models import ensure_assets, load_backend, checked_detection
    path = tmp_path / 'facial_recognition.yml'
    path.write_text(f'version: 1\nmodels: {{detector: yunet, recognizer: {recognizer}, device: cpu}}\n', encoding='utf-8')
    config = load(path)
    backend = load_backend(config, ensure_assets(config))
    try:
        with Image.new('RGB', (128, 128), (100, 120, 140)) as image:
            faces = backend.detect(image)
            assert isinstance(faces, list)
            detection = checked_detection([0, 0, 128, 128, 38, 44, 88, 44, 64, 66, 42, 90, 86, 90, .99])
            embedding = backend.embed(image, detection)
            assert embedding.shape == (dimensions,)
            assert backend.device == 'cpu'
            with backend.prepared_frame(image):
                assert [face.row for face in backend.detect(image)] == [face.row for face in faces]
                assert np.allclose(backend.embed(image, detection), embedding, atol=1e-7)
            assert backend._prepared_frame is None
    finally:
        backend.close()


@pytest.mark.skipif(os.getenv('STASH_AI_RECOGNITION_SMOKE') != '1', reason='explicit pinned face-model smoke')
def test_ort_contract_matches_cpu_sface_and_handles_non_square_yunet(tmp_path):
    import cv2
    import numpy as np
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.models import ensure_assets, load_backend, checked_detection
    from stash_ai.recognition.runtime import runtime
    from stash_ai.recognition.onnx_models import OnnxYuNet, OnnxSFace
    from tests.helpers.recognition import textured_image
    from dataclasses import replace
    from stash_ai.recognition.config import plain
    original = load(root=tmp_path, device='cpu')
    data = plain(original.data)
    data['models']['detector'] = 'yunet'
    config = replace(original, data=data)
    paths = ensure_assets(config)
    cpu = load_backend(config, paths)
    detector = OnnxYuNet(config, paths['yunet'], cv2, runtime(config))
    recognizer = OnnxSFace(config, paths['sface'], runtime(config))
    try:
        for shape in ((128,128), (160,90)):
            with Image.new('RGB',shape,(100,120,140)) as image:
                assert isinstance(detector.detect(cpu._bgr(image)),list)
        face = checked_detection([0,0,128,128,38,44,88,44,64,66,42,90,86,90,.99])
        with textured_image() as image:
            aligned = cpu.recognizer.alignCrop(cpu._bgr(image), np.asarray(face.row,dtype=np.float32))
            reference = cpu.embed(image,face)
            converted = recognizer.embed(aligned)
            assert np.allclose(reference,converted,atol=2e-5)
            assert float(np.dot(reference,converted)) > .99999
    finally:
        cpu.close()
        detector.close()
        recognizer.close()


@pytest.mark.skipif(os.getenv('STASH_AI_RECOGNITION_SMOKE') != '1', reason='explicit pinned visibility-model smoke')
@pytest.mark.parametrize('region_mode', ['mesh', 'ellipse'])
def test_faceocc_checkpoint_detects_lower_face_covering_on_public_domain_portrait(tmp_path, region_mode):
    from contextlib import ExitStack
    import hashlib
    from io import BytesIO
    import requests
    from PIL import Image, ImageDraw
    from stash_ai.recognition.config import load
    from stash_ai.recognition.models import checked_detection, ensure_assets
    from stash_ai.recognition.visibility import FaceVisibility, prepare_measurement
    from stash_ai.recognition.face_contour import resolve
    from dataclasses import replace
    from tests.test_recognition_visibility_rotation import quarter_turn
    # Grace Hopper, US Navy/public domain; immutable content checked independently
    # of the matplotlib tag. Download only in the explicit model smoke jobs.
    url = 'https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.1/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg'
    with requests.get(url, timeout=30) as response:
        response.raise_for_status()
        data = response.content
    assert hashlib.sha256(data).hexdigest() == 'a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130'
    config = load(root=tmp_path, device='cpu')
    config = replace(config, data=dict(config.data, detection=dict(config.data['detection'], visibility_region=region_mode)))
    model = FaceVisibility(config, ensure_assets(config, keys=['faceocc']))
    face = checked_detection([182.86642, 120.47789, 170.25731, 212.48526, 227.11728, 193.32774,
                              307.84845, 195.3718, 266.4793, 240.008, 228.50974, 267.86563,
                              302.18546, 269.56985, .94316435])
    try:
        with Image.open(BytesIO(data)) as original, original.convert('RGB') as image, image.copy() as covered:
            x, y, width, height = face.box
            ImageDraw.Draw(covered).rectangle((x, y + height * .6, x + width, y + height), fill=(70, 90, 160))
            visible = model.measure(image, face)
            occluded = model.measure(covered, face)
            prepared = prepare_measurement(covered, face, 0., region_mode=region_mode)
            selected = resolve(model, prepared)
            blob, region, total = prepared[0], selected.mask, selected.total
            assert visible['visibility']['region_method'] == region_mode
            assert occluded['visibility']['region_method'] == region_mode
            assert total == occluded['visibility']['face_pixels']
            assert int(model.inspect_foreground(blob, region).sum()) == occluded['visibility']['visible_pixels']
            batched = model.measure_many([(image,face),(covered,face)])
            assert [v['visibility']['face_pixels'] for v in batched]==[visible['visibility']['face_pixels'],occluded['visibility']['face_pixels']]
            assert [v['visible_face_fraction'] for v in batched]==pytest.approx(
                [visible['visible_face_fraction'],occluded['visible_face_fraction']],abs=.001)
            with ExitStack() as images:
                rotated = []
                for turns in (1, 2, 3):
                    for source in (image, covered):
                        pixels, detection = quarter_turn(source, face, turns)
                        rotated.append((images.enter_context(pixels), detection))
                results = model.measure_many(rotated)
            for uncovered, hidden in zip(results[::2], results[1::2]):
                assert uncovered['visible_face_fraction'] > .8
                assert hidden['visible_face_fraction'] < .8
                assert uncovered['visible_face_fraction']-hidden['visible_face_fraction'] > .2
                assert uncovered['visible_face_fraction'] == pytest.approx(visible['visible_face_fraction'], abs=.04)
                assert hidden['visible_face_fraction'] == pytest.approx(occluded['visible_face_fraction'], abs=.04)
        assert visible['visible_face_fraction'] > .8
        assert occluded['visible_face_fraction'] < .8
        assert visible['visible_face_fraction'] - occluded['visible_face_fraction'] > .2
        # Occlusion must not shrink the denominator along with the visible mask.
        assert occluded['visibility']['face_pixels'] == pytest.approx(
            visible['visibility']['face_pixels'], rel=.05 if region_mode == 'mesh' else 0)
        assert model.device == 'cpu'
    finally:
        model.close()


@pytest.mark.skipif(os.getenv('STASH_AI_RECOGNITION_SMOKE') != '1', reason='explicit pinned expression-model smoke')
def test_pinned_expression_task_associates_mesh_and_scores_public_domain_portrait(tmp_path):
    import hashlib
    from io import BytesIO
    import requests
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.expression import FaceExpression, cached
    from stash_ai.recognition.models import checked_detection, ensure_assets
    with requests.get('https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.1/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg', timeout=30) as response:
        response.raise_for_status(); data = response.content
    assert hashlib.sha256(data).hexdigest() == 'a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130'
    path = tmp_path / 'expression.yml'
    path.write_text('version: 1\nmodels: {device: cpu}\nportraits: {weights: {closed_mouth: 2}}\n')
    config = load(path)
    model = FaceExpression(config, ensure_assets(config))
    face = checked_detection([182.86642, 120.47789, 170.25731, 212.48526, 227.11728, 193.32774,
                              307.84845, 195.3718, 266.4793, 240.008, 228.50974, 267.86563,
                              302.18546, 269.56985, .94316435])
    try:
        with Image.open(BytesIO(data)) as image:
            result = model.measure(image, face)
        assert cached(result) and result['expression']['device'] == 'cpu'
        assert result['closed_mouth'] > .8 and result['open_eyes'] > .5
        assert result['open_mouth'] + result['closed_mouth'] == pytest.approx(1)
    finally:
        model.close()


@pytest.mark.skipif(os.getenv('STASH_AI_RECOGNITION_SMOKE') != '1', reason='explicit pinned ONNX detector smoke')
@pytest.mark.parametrize('detector_key', ['scrfd_10g', 'scrfd_2_5g', 'retinaface_r50'])
def test_pinned_scrfd_onnx_kps_cpu_smoke(tmp_path, detector_key):
    """Run the actual pinned 640px model, not just synthetic decoder fixtures."""
    import numpy as np
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.models import ensure_assets, YuNetDetector

    yaml = tmp_path / 'scrfd-smoke.yml'
    yaml.write_text(f'version: 1\nmodels: {{detector: {detector_key}, device: cpu}}\n')
    config = load(yaml)
    assets = ensure_assets(config, keys=[detector_key])
    backend = YuNetDetector(config, assets)
    try:
        assert backend.detector_device == 'cpu'
        assert backend.provenance['detector']['detector'] == detector_key
        for width, height in ((160, 160), (160, 90)):
            with Image.new('RGB', (width, height), (110, 100, 90)) as image:
                with backend.prepared_frame(image):
                    detections = backend.detect(image)
                    assert isinstance(detections, list)
                    assert all(len(detection.row) == 15 and np.isfinite(detection.row).all()
                               for detection in detections)
    finally:
        backend.close()


@pytest.mark.skipif(os.getenv('STASH_AI_RECOGNITION_SMOKE') != '1', reason='explicit pinned RetinaFace image smoke')
def test_pinned_retinaface_landmarks_on_public_domain_portrait(tmp_path):
    import hashlib
    from io import BytesIO
    import requests
    import numpy as np
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.models import ensure_assets, YuNetDetector
    from stash_ai.recognition.quality import alignment_measurements
    with requests.get('https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.1/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg', timeout=30) as response:
        response.raise_for_status();data=response.content
    assert hashlib.sha256(data).hexdigest()=='a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130'
    path=tmp_path/'retinaface.yml'
    path.write_text('version: 2\nface: {detector: {backend: retinaface_r50, score_threshold: 0.55}}\nexecution: {gpu: {device: cpu}}\n')
    config=load(path)
    from dataclasses import replace
    from stash_ai.recognition.config import plain
    settings=plain(config.data)
    settings['detection']['pose_guided_rotation']=True
    settings['sapiensid']['enabled']=True
    settings['sapiensid']['collection']['enabled']=True
    config=replace(config,data=settings)
    backend=YuNetDetector(config,ensure_assets(config,keys=['retinaface_r50']))
    try:
        with Image.open(BytesIO(data)) as image:
            faces=backend.detect(image)
            assert len(faces)==1
            assert alignment_measurements(faces[0],image)['alignment_valid']
            # A genuine positive sample catches channel, anchor, scale and
            # output-order errors that a blank-image smoke cannot detect.
            reference=np.array([[227,193],[308,195],[266,240],[229,268],[302,270]])
            assert np.max(np.linalg.norm(np.asarray(faces[0].landmarks)-reference,axis=1))<20
            from stash_ai.recognition.rotated_faces import restore
            upright=faces[0]
            for transpose,ccw in ((Image.Transpose.ROTATE_90,True),(Image.Transpose.ROTATE_270,False)):
                with image.transpose(transpose) as turned:
                    faces=backend.detect(turned)
                    assert len(faces)==1 and alignment_measurements(faces[0],turned)['alignment_valid']
                    restored=restore(faces[0],(0,0,*image.size),ccw)
                    assert np.max(np.linalg.norm(np.asarray(restored.landmarks)-upright.landmarks,axis=1))<20
    finally:backend.close()


@pytest.mark.skipif(os.getenv('STASH_AI_RECOGNITION_SMOKE') != '1', reason='explicit pinned facebox-landmark smoke')
@pytest.mark.parametrize('key', ['scrfd_10g', 'scrfd_2_5g', 'retinaface_r50', 'yunet'])
def test_pinned_facebox_landmarks_preserve_box_on_public_domain_portrait(tmp_path, key):
    import hashlib
    from io import BytesIO
    import requests
    import numpy as np
    from PIL import Image
    from stash_ai.recognition.config import load
    from stash_ai.recognition.models import ensure_assets, YuNetDetector
    from stash_ai.recognition.quality import alignment_measurements
    from stash_ai.recognition.landmark_repair_audit import capture
    with requests.get('https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.1/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg', timeout=30) as response:
        response.raise_for_status(); data = response.content
    assert hashlib.sha256(data).hexdigest() == 'a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130'
    path = tmp_path/'facebox.yml'
    path.write_text(f'version: 2\nface: {{detector: {{backend: {key}, score_threshold: 0.55, landmarks_from_facebox: true}}}}\nexecution: {{gpu: {{device: cpu}}}}\n')
    config = load(path)
    backend = YuNetDetector(config, ensure_assets(config, keys=[key]))
    try:
        with Image.open(BytesIO(data)) as image, capture() as reports:
            faces = backend.detect(image)
            assert len(faces) == 1 and alignment_measurements(faces[0], image)['alignment_valid']
            trace = reports[0]['faces'][0]
            assert trace['status'] == 'refined_from_facebox'
            assert trace['result']['box'] == trace['original']['box']
            assert trace['result']['score'] == trace['original']['score']
            assert trace['result']['landmarks'] != trace['original']['landmarks']
            # Approximate visible landmark locations catch coordinate/channel
            # errors. This frontal portrait is not an extreme-profile benchmark.
            reference = np.array([[227,193],[308,195],[266,240],[229,268],[302,270]])
            assert np.max(np.linalg.norm(np.asarray(faces[0].landmarks)-reference, axis=1)) < 20
    finally:
        backend.close()
