"""Typed identity boundaries with adapters preserving historical signature JSON."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PredictiveIdentity:
    canonical_json: str

    def legacy(self):
        return json.loads(self.canonical_json)


@dataclass(frozen=True)
class ExecutionIdentity:
    batch_size: int
    asset_paths: tuple[str, ...]
    prefetch_batches: int = 0
    result_prefetch_batches: int = 32
    checkpoint_batches: int = 8
    prepare_workers: int = 1

    @classmethod
    def from_legacy(cls, value, args=None):
        execution = {name: getattr(args, name) for name in (
            'prefetch_batches', 'result_prefetch_batches', 'checkpoint_batches', 'prepare_workers')} if args is not None else {}
        return cls(value['batch_size'], tuple(path for path, _ in value['files']), **execution)


@dataclass(frozen=True)
class ProvenanceIdentity:
    canonical_json: str

    @classmethod
    def from_legacy(cls, value):
        return cls(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')))

    def legacy(self):
        return json.loads(self.canonical_json)

    def predictive(self, *, checkpoint=False):
        value = self.legacy()
        excluded = {'batch_size', '_input_options', '_signature_proof'}
        if not checkpoint:
            excluded.update(('retention', 'scene_threshold', 'marker_threshold'))
        result = {key: item for key, item in value.items() if key not in excluded}
        result['files'] = [[Path(path).name, checksum] for path, checksum in value['files']]
        return PredictiveIdentity(json.dumps(result, sort_keys=True, ensure_ascii=False,
                                            allow_nan=False, separators=(',', ':')))
