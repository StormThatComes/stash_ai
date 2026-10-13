"""Versioned, immutable face-only configuration and semantic stage identities."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType


def defaults():
    # Conservative starting policies; these are not calibrated probabilities or
    # claims about accuracy on an operator's VTT thumbnails.
    return {
        'version': 1,
        'models': {'detector': 'yunet', 'recognizer': 'sface', 'device': 'cuda',
                   'directory': 'recognition_models', 'download_missing': True},
        'detection': {'score_threshold': .9, 'nms_threshold': .3,
                      'min_face_pixels': 32, 'min_sharpness': 10., 'max_faces_per_frame': 100},
        'recognition': {'metric': 'cosine', 'match_threshold': .65,
                        'ambiguity_margin': .1, 'min_support': 3},
        'clustering': {'cohesion_threshold': .60, 'max_samples_per_person': 32,
                       'max_people_per_lookup': 20000, 'max_reference_bytes': 134217728},
        'portraits': {'enabled': True, 'source': 'vtt', 'min_face_pixels': 96, 'min_sharpness': 25.,
                     'max_pose_deviation': .35, 'min_exposure': 30., 'max_exposure': 225.,
                     'max_clipped_fraction': .15, 'margin': .35, 'aspect_ratio': None,
                     'jpeg_quality': 90, 'min_visible_face_fraction': .8, 'max_candidates': 8,
                     'time_offsets': [0., -.5, .5], 'timeout_seconds': 30.,
                     'max_video_bytes': 67108864, 'ffmpeg': 'ffmpeg',
                     'weights': {'frontal': 3., 'sharpness': 2., 'resolution': 1., 'exposure': 1., 'visibility': 3.,
                                 'neutral_expression': 0., 'closed_mouth': 0., 'open_mouth': 0., 'open_eyes': 0.}},
        'execution': {'max_cues': 0, 'checkpoint_cues': 32, 'timeout_seconds': 120.},
        'storage': {'database': 'faces_state.sqlite3', 'max_crop_bytes': 1048576},
        'names': {'file': None},
    }


def plain(value):
    if isinstance(value, (dict, MappingProxyType)):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    return tuple(_freeze(item) for item in value) if isinstance(value, list) else value


def parse_yaml(raw, *, context='Facial recognition', max_bytes=131072):
    import yaml
    if len(raw) > max_bytes:
        raise ValueError(f'{context} YAML exceeds the byte limit.')
    text = raw.decode('utf-8-sig')
    depth = count = 0
    for event in yaml.parse(text):
        count += 1
        if isinstance(event, yaml.AliasEvent):
            raise ValueError(f'{context} YAML aliases are not supported.')
        if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
            depth += 1
        if isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
            depth -= 1
        if count > 20000 or depth > 16:
            raise ValueError(f'{context} YAML exceeds the structure limit.')

    class Loader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            result = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str) or key in result:
                    raise ValueError(f'{context}: duplicate/non-string YAML key; quote IDs.')
                result[key] = self.construct_object(value_node, deep=deep)
            return result

    return yaml.load(text, Loader=Loader)


def _merge(base, supplied, path=''):
    if not isinstance(supplied, dict) or set(supplied) - set(base):
        raise ValueError(f'{path or "configuration"}: unknown keys or expected mapping.')
    for key, value in supplied.items():
        name = f'{path}.{key}' if path else key
        if isinstance(base[key], dict):
            _merge(base[key], value, name)
        else:
            base[key] = value


def _number(value, name, low, high, *, integer=False):
    valid = type(value) is int if integer else type(value) in (int, float)
    if not valid or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name}: expected {"integer" if integer else "finite number"} in [{low}, {high}].')


def validate(data):
    if type(data['version']) is not int or data['version'] != 1:
        raise ValueError('facial_recognition.yml version must be 1.')
    models = data['models']
    if models['detector'] != 'yunet' or models['recognizer'] not in ('sface', 'auraface'):
        raise ValueError('Supported detector/recognizer: yunet / sface or auraface.')
    if models['device'] not in ('auto', 'cpu', 'cuda') or type(models['download_missing']) is not bool:
        raise ValueError('Invalid face device/download policy.')
    for section, names in {
        'detection': ('score_threshold', 'nms_threshold'),
        'recognition': ('match_threshold', 'ambiguity_margin'),
        'clustering': ('cohesion_threshold',),
        'portraits': ('max_pose_deviation', 'max_clipped_fraction', 'margin', 'min_visible_face_fraction'),
    }.items():
        for key in names:
            _number(data[section][key], f'{section}.{key}', 0, 1)
    if data['recognition']['metric'] != 'cosine':
        raise ValueError('Face recognition uses cosine similarity (higher means more similar).')
    if data['clustering']['cohesion_threshold'] > data['recognition']['match_threshold']:
        raise ValueError('clustering.cohesion_threshold must not exceed recognition.match_threshold.')
    integers = {
        'detection': {'min_face_pixels': (8, 4096), 'max_faces_per_frame': (1, 1000)},
        'recognition': {'min_support': (2, 1000)},
        'clustering': {'max_samples_per_person': (2, 128), 'max_people_per_lookup': (1, 100000),
                       'max_reference_bytes': (1048576, 1073741824)},
        'portraits': {'min_face_pixels': (8, 4096), 'max_candidates': (1, 64), 'jpeg_quality': (1, 100),
                      'max_video_bytes': (1048576, 1073741824)},
        'execution': {'max_cues': (0, 10000), 'checkpoint_cues': (1, 1000)},
        'storage': {'max_crop_bytes': (1024, 8388608)},
    }
    for section, fields in integers.items():
        for key, bounds in fields.items():
            _number(data[section][key], f'{section}.{key}', *bounds, integer=True)
    for section in ('detection', 'portraits'):
        _number(data[section]['min_sharpness'], f'{section}.min_sharpness', 0, 100000)
    for section in ('execution', 'portraits'):
        _number(data[section]['timeout_seconds'], f'{section}.timeout_seconds', 1, 600)
    portrait = data['portraits']
    if type(portrait['enabled']) is not bool:
        raise ValueError('portraits.enabled must be boolean.')
    if portrait['source'] not in ('video', 'vtt'):
        raise ValueError('portraits.source must be video or vtt.')
    if portrait['aspect_ratio'] is not None:
        _number(portrait['aspect_ratio'], 'portraits.aspect_ratio', .1, 10)
    for key in ('min_exposure', 'max_exposure'):
        _number(portrait[key], f'portraits.{key}', 0, 255)
    if portrait['min_exposure'] >= portrait['max_exposure']:
        raise ValueError('Portrait exposure range is empty.')
    offsets = portrait['time_offsets']
    if not isinstance(offsets, list) or not 1 <= len(offsets) <= 9:
        raise ValueError('portraits.time_offsets requires 1..9 offsets in seconds.')
    for offset in offsets:
        _number(offset, 'portraits.time_offsets', -5, 5)
    for key, value in portrait['weights'].items():
        _number(value, f'portraits.weights.{key}', 0, 100)
    if not sum(portrait['weights'].values()):
        raise ValueError('Portrait ranking weights cannot all be zero.')
    for section, key in (('models', 'directory'), ('storage', 'database'), ('portraits', 'ffmpeg')):
        value = data[section][key]
        if not isinstance(value, str) or not value.strip() or '\x00' in value:
            raise ValueError(f'{section}.{key} must be a nonempty path.')
    if data['names']['file'] is not None and (not isinstance(data['names']['file'], str)
                                             or not data['names']['file'].strip()):
        raise ValueError('names.file must be a path or null.')


@dataclass(frozen=True)
class FaceConfig:
    data: object
    source: bytes
    source_path: Path | None
    database: Path
    model_dir: Path
    names_path: Path | None

    def snapshot(self):
        baseline = defaults()
        custom = any(plain(self.data[section]) != baseline[section]
                     for section in ('detection', 'recognition', 'clustering', 'portraits'))
        return {'version': 1, 'source_sha256': hashlib.sha256(self.source).hexdigest(),
                'source': self.source.decode('utf-8-sig'), 'effective': plain(self.data),
                'paths': {'database': str(self.database), 'models': str(self.model_dir),
                          'names': str(self.names_path) if self.names_path else None},
                'policy_origin': 'custom' if custom else 'bundled-default',
                'default_policy': ('custom-operator-policy-not-domain-calibrated' if custom
                                   else 'conservative-v1-not-domain-calibrated')}

    def identity(self, *sections):
        return digest({name: self.data[name] for name in sections})


def load(path=None, *, database=None, device=None, root=None):
    root = Path.cwd() if root is None else Path(root)
    selected = Path(path).expanduser().resolve() if path is not None else root / 'facial_recognition.yml'
    if path is not None or selected.is_file():
        # Read bounded bytes before parsing, not an unbounded read_bytes().
        with selected.open('rb') as stream:
            raw = stream.read(131073)
        supplied = parse_yaml(raw)
        if not isinstance(supplied, dict) or 'version' not in supplied:
            raise ValueError('facial_recognition.yml requires a versioned mapping.')
        base = selected.parent
    else:
        raw, supplied, base, selected = b'', {}, root.resolve(), None
    data = defaults()
    _merge(data, supplied)
    if device is not None:
        data['models']['device'] = device
    validate(data)
    def resolve(value):
        p = Path(value).expanduser()
        return (base / p).resolve() if not p.is_absolute() else p.resolve()
    return FaceConfig(_freeze(data), raw, selected,
                      Path(database).expanduser().resolve() if database is not None else resolve(data['storage']['database']),
                      resolve(data['models']['directory']),
                      resolve(data['names']['file']) if data['names']['file'] else None)
