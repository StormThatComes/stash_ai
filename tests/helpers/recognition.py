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

    def close(self):
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
        def timestamp(i):
            return f'{i//3600:02d}:{i//60%60:02d}:{i%60:02d}.000'
        vtt = 'WEBVTT\n\n' + '\n\n'.join(
            f'{timestamp(i)} --> {timestamp(i+1)}\nsprite.jpg#xywh=0,0,160,160' for i in range(count))
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


class Visibility:
    """Synthetic visibility only; real model behavior has a separate opt-in smoke."""
    def __init__(self, *args):
        self.calls, self.closed = 0, False

    def measure(self, image, face):
        from stash_ai.recognition.visibility import CONTRACT_ID
        self.calls += 1
        return {'visible_face_fraction': 1., 'visibility': {'contract': CONTRACT_ID, 'device': 'cpu'}}

    def close(self):
        self.closed = True


class Stash:
    """Fake server preserves manual links and can lose mutation responses."""
    url = 'http://stash/'
    api_key = ''

    def __init__(self, **kwargs):
        self.profiles = {'77': {'id': '77', 'name': 'Manual', 'details': '', 'image_path': ''}}
        self.links = {'1': ['77'], '2': []}
        self.media_calls, self.writes, self.reads = [], [], []
        self.fail_create = self.fail_link = False
        self.fail_merge = False
        self.merge_visible = True
        self.create_visible = True
        self.images = {}
        self.media = SimpleNamespace(get=self.get)
        self.closed = False

    def get(self, url, **kwargs):
        self.media_calls.append(url)
        if '/performer/' in url:
            identifier = url.split('/performer/')[1].split('/')[0]
            return Response(self.images[identifier])
        identifier = url.rsplit('/', 1)[-1].split('.')[0]
        return MediaClient(identifier).get(url, **kwargs)

    def scene(self, identifier):
        self.reads.append(('scene', identifier))
        return {'id': identifier, 'paths': {'vtt': f'/vtt/{identifier}.vtt',
                'sprite': f'/sprite/{identifier}.jpg', 'stream': f'/scene/{identifier}/stream'},
                'performers': [{'id': p} for p in self.links.setdefault(identifier, [])]}

    def scene_ids(self):
        return iter(sorted(self.links))

    def new_media_reader(self):
        return SimpleNamespace(url=self.url, api_key=self.api_key, scene=self.scene,
                               media=SimpleNamespace(get=self.media.get), close=lambda: None)

    def performer(self, identifier):
        self.reads.append(('performer', identifier))
        return dict(self.profiles[identifier])

    def lookup_performer(self, identifier):
        self.reads.append(('lookup_performer', identifier))
        return dict(self.profiles[identifier]) if identifier in self.profiles else None

    def merge_person(self, source, destination, details):
        self.writes.append(('merge', source, destination))
        if self.merge_visible:
            for scene, links in self.links.items():
                self.links[scene] = list(dict.fromkeys(destination if p == source else p for p in links))
            self.profiles[destination]['details'] = details
            del self.profiles[source]
            self.images.pop(source, None)
        if self.fail_merge:
            self.fail_merge = False
            raise TimeoutError('lost merge response')
        return self.performer(destination)

    def find_token(self, token):
        return [dict(p) for p in self.profiles.values() if token in p['details'].splitlines()]

    def create_person(self, name, token):
        self.writes.append(('create', name))
        value = {'id': str(100 + len(self.profiles)), 'name': name, 'details': token, 'image_path': ''}
        if self.create_visible:
            self.profiles[value['id']] = value
        if self.fail_create:
            self.fail_create = False
            raise TimeoutError('lost create response')
        return dict(value)

    def add_scene_person(self, scene, performer):
        self.writes.append(('link', scene, performer))
        if performer not in self.links[scene]:
            self.links[scene].append(performer)
        if self.fail_link:
            self.fail_link = False
            raise TimeoutError('lost link response')
        return self.scene(scene)

    def update_person(self, performer, **fields):
        self.writes.append(('metadata', performer, fields))
        if 'image' in fields:
            import base64
            image = fields.pop('image')
            if image == '':
                self.images.pop(performer, None)
                fields['image_path'] = self.url + f'performer/{performer}/image?t=123&default=true'
            else:
                self.images[performer] = base64.b64decode(image.split(',', 1)[1])
                fields['image_path'] = self.url + f'performer/{performer}/image?t=123'
        self.profiles[performer].update(fields)
        return self.performer(performer)

    def remove_scene_person(self, scene, performer):
        self.writes.append(('unlink', scene, performer))
        self.links[scene] = [p for p in self.links[scene] if p != performer]
        return self.scene(scene)

    def close(self):
        self.closed = True
