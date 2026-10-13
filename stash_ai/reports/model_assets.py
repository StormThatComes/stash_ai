"""Offline model-asset health report: deterministic local integrity observations."""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from stash_ai.backends.vocabulary import metadata_files, weight_files
from stash_ai.config import load_model_settings
from stash_ai.reports.io import write_json_report


def _identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _file_status(path):
    observed = False
    try:
        initial = path.stat()
        observed = True
        if not stat.S_ISREG(initial.st_mode):
            return dict(status='missing')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            before = os.fstat(stream.fileno())
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        current = path.stat()
        identity = _identity(before)
        if any(_identity(value) != identity for value in (initial, after, current)):
            return dict(status='changed_during_scan')
        if not before.st_size:
            return dict(status='empty', bytes=0)
        return dict(status='present_unverified', bytes=before.st_size, sha256=digest.hexdigest(), _identity=identity)
    except FileNotFoundError:
        return dict(status='changed_during_scan' if observed else 'missing')
    except OSError as error:
        return dict(status='read_error', error=type(error).__name__)


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
            # Hash the small metadata again after parsing, including same-size
            # replacements with preserved mtime. Weight files are not reread.
            for name in metadata_files(setting.source):
                current = _file_status(setting.model_dir / name)
                if current['status'] == 'read_error':
                    records[name] = current
                    meta_state, labels = 'read_error', None
                elif (current['status'] != 'present_unverified'
                        or current.get('sha256') != records[name].get('sha256')
                        or current.get('_identity') != records[name].get('_identity')):
                    records[name]['status'] = 'changed_during_scan'
                    meta_state, labels = 'changed_during_scan', None
        else:
            labels = None
        # Detect replacement/removal during the rest of this model's scan too,
        # without doubling reads of potentially large weight files.
        for name, record in records.items():
            if record['status'] != 'present_unverified':
                continue
            try:
                if _identity((setting.model_dir / name).stat()) != record['_identity']:
                    record['status'] = 'changed_during_scan'
                    if name in metadata_files(setting.source):
                        meta_state, labels = 'changed_during_scan', None
            except FileNotFoundError:
                record['status'] = 'changed_during_scan'
                if name in metadata_files(setting.source):
                    meta_state, labels = 'changed_during_scan', None
            except OSError as error:
                record.update(status='read_error', error=type(error).__name__)
                if name in metadata_files(setting.source):
                    meta_state, labels = 'read_error', None
        public_records = {name: {key: value for key, value in record.items() if not key.startswith('_')}
                          for name, record in records.items()}
        models.append(dict(model=setting.source.key, revision=setting.source.revision,
                           files=public_records, metadata=meta_state, labels=labels,
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
