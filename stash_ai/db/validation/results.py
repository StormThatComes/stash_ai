"""Validate saved evidence, derived indexes and effective thresholds."""
from __future__ import annotations

import json
import zlib

from ..common import count_rows
from ..payloads import valid_result
from .common import decoded_payload, tick


def check_results_and_index(db, tables, errors, progress=None):
    if "results" not in tables:
        return 0

    indexed_signature = None
    checked = 0
    query = ('SELECT r.kind,r.data,i.populated,i.signature,t.threshold,m.format,m.data '
             'FROM results AS r LEFT JOIN result_index AS i USING(scope,scene,model,kind) '
             'LEFT JOIN result_thresholds AS t USING(scope,scene,model,kind) '
             'LEFT JOIN result_metadata AS m USING(scope,scene,model,kind)')
    for kind, blob, populated, signature, threshold, metadata_format, metadata in db.execute(query):
        checked += 1
        tick(progress)
        try:
            record = decoded_payload(blob, dict)
            if not valid_result(record, kind):
                raise ValueError('Invalid result record.')
            from stash_ai.evidence.formats import validate_current
            validate_current(record)
            if {'publication_threshold', 'scene_threshold_override', 'marker_threshold_override', 'refinement'} & record.keys():
                raise ValueError('Noncanonical result record.')
            if metadata_format is not None:
                from stash_ai.storage.metadata import FORMAT, decode, encode, summarize
                try:
                    if metadata_format != FORMAT:
                        raise ValueError('Unsupported result metadata format.')
                    if encode(decode(metadata)) != encode(summarize(record)):
                        errors['result_metadata_mismatch'] += 1
                except (TypeError, ValueError):
                    errors['result_metadata_invalid'] += 1
            if threshold is not None:
                from stash_ai.storage.thresholds import overlay
                try:
                    record = overlay(dict(record), kind, threshold)
                except (TypeError, ValueError):
                    pass
            expected_populated = int(bool(record['items']))
            if indexed_signature is None:
                from stash_ai.evidence.thresholds import indexed_signature as signature_for
                indexed_signature = signature_for
            expected_signature = indexed_signature(record)
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError, zlib.error):
            errors["result_payload_invalid"] += 1
            continue

        if populated is None and "result_index" in tables:
            errors["result_missing_index"] += 1
        elif (populated not in (0, 1)
              or populated != expected_populated
              or signature != expected_signature):
            errors["result_index_mismatch"] += 1

    if "result_index" in tables:
        errors["result_index_orphan"] += db.execute("""
            SELECT COUNT(*) FROM result_index AS i
            WHERE NOT EXISTS (
                SELECT 1 FROM results AS r
                WHERE r.scope=i.scope AND r.scene=i.scene
                  AND r.model=i.model AND r.kind=i.kind
            )
        """).fetchone()[0]
    if 'result_metadata' in tables:
        errors['result_metadata_orphan'] += db.execute('''
            SELECT COUNT(*) FROM result_metadata AS m WHERE NOT EXISTS (
                SELECT 1 FROM results AS r WHERE r.scope=m.scope AND r.scene=m.scene
                    AND r.model=m.model AND r.kind=m.kind)
        ''').fetchone()[0]
    return checked


def check_thresholds(db, tables, errors):
    if 'result_thresholds' not in tables:
        return
    from stash_ai.storage.thresholds import validate
    for data, in db.execute('SELECT threshold FROM result_thresholds'):
        try:
            validate(data)
        except (TypeError, ValueError):
            errors['result_threshold_invalid'] += 1
    if 'results' in tables:
        errors['result_threshold_orphan'] += db.execute('''
            SELECT COUNT(*) FROM result_thresholds AS t WHERE NOT EXISTS (
                SELECT 1 FROM results AS r WHERE r.scope=t.scope AND r.scene=t.scene
                    AND r.model=t.model AND r.kind=t.kind)
        ''').fetchone()[0]
    else:
        errors['result_threshold_orphan'] += count_rows(db, tables, 'result_thresholds')
