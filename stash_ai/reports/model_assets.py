"""Offline model-asset health report: deterministic local integrity observations."""
from __future__ import annotations

import hashlib
from pathlib import Path

from stash_ai.backends.vocabulary import metadata_files, weight_files
from stash_ai.config import load_model_settings
from stash_ai.reports.io import write_json_report


def _file_status(path):
    if not path.is_file():
        return dict(status='missing')
    size = path.stat().st_size
    if size == 0:
        return dict(status='empty', bytes=0)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(status='present_unverified', bytes=size, sha256=digest.hexdigest())


def build_report(settings):
    models = []
    for setting in settings:
        names = tuple(dict.fromkeys((*metadata_files(setting.source), *weight_files(setting))))
        records = {name: _file_status(setting.model_dir / name) for name in names}
        meta_state = 'not_checked'
        if all(records[name]['status'] == 'present_unverified' for name in metadata_files(setting.source)):
            # Local-only parsing of vocabulary and metadata, never load model weights.
            try:
                from stash_ai.backends.vocabulary import load_vocabulary
                from dataclasses import replace
                vocabulary = load_vocabulary(replace(setting, download_missing=False))
                meta_state = 'valid'
                labels = len(vocabulary.labels)
            except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
                meta_state = 'invalid_' + type(exc).__name__
                labels = None
        else:
            labels = None
        models.append(dict(model=setting.source.key, revision=setting.source.revision,
                           files=records, metadata=meta_state, labels=labels,
                           healthy=meta_state == 'valid' and all(
                               value['status'] == 'present_unverified' for value in records.values())))
    return dict(format=1, mode='offline-model-assets', models=models,
                note='A local SHA256 identifies bytes but is not an upstream authenticity proof; '
                     'non-empty weight files without pinned expected digests remain unverified.')


def run(app, args):
    settings = load_model_settings(args.model_config, args.models, None, offline=True)
    report = build_report(settings)
    destination = write_json_report(report, args.model_assets_report)
    print(f"Model asset health report written to {destination}. No inference backends were loaded.")
    for model in report['models']:
        print(f"  {model['model']}: {'files present' if model['healthy'] else 'incomplete or invalid'}"
              f" (metadata={model['metadata']})")
    return int(not all(model['healthy'] for model in report['models']))
