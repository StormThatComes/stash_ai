"""Synthetic face geometry/embeddings and in-memory selected-scene media."""
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from PIL import Image

from stash_ai.recognition.models import checked_detection


FACE = checked_detection([16, 16, 128, 128, 49, 60, 111, 60, 80, 84, 56, 108, 104, 108, .99])


def textured_image():
    grid = np.indices((160, 160)).sum(axis=0) % 2
    pixels = np.repeat((grid[..., None] * 210 + 20).astype('uint8'), 3, axis=2)
    return Image.fromarray(pixels)


class Response:
    status_code = 200
    headers = {}

    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        for start in range(0, len(self.data), size):
            yield self.data[start:start + size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MediaClient:
    url = 'http://stash/'

    def __init__(self, scene_id='1', count=3):
        self.calls = []
        self.scene = {'id': scene_id, 'paths': {'vtt': f'/vtt/{scene_id}.vtt', 'sprite': f'/sprite/{scene_id}.jpg'}}
        with textured_image() as image:
            output = BytesIO()
            image.save(output, format='PNG')
            sprite = output.getvalue()
        vtt = 'WEBVTT\n\n' + '\n\n'.join(
            f'00:00:{i:02d}.000 --> 00:00:{i+1:02d}.000\nsprite.jpg#xywh=0,0,160,160' for i in range(count))
        self.data = {self.url + f'vtt/{scene_id}.vtt': vtt.encode(),
                     self.url + f'sprite/{scene_id}.jpg': sprite}
        self.media = SimpleNamespace(get=self.get)

    def get(self, url, **kwargs):
        self.calls.append(url)
        return Response(self.data[url])


class Backend:
    dimensions = 4
    device = 'cpu'

    def __init__(self, *, fail_at=None, vector=(1., 0., 0., 0.)):
        self.calls = 0
        self.closed = False
        self.fail_at = fail_at
        self.vector = vector

    def detect(self, image):
        return [FACE]

    def embed(self, image, face):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('synthetic inference failed')
        return np.asarray(self.vector, dtype='float32')

    def close(self):
        self.closed = True
