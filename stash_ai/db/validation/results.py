"""Validate saved evidence, derived indexes and effective thresholds."""
from __future__ import annotations

import json
import zlib

from ..common import count_rows
from ..payloads import valid_result_segments
from .common import decoded_payload, tick


def check_results_and_index(db, tables, errors, progress=None):
    if "results" not in tables:
        return 0

    indexed_signature = None
    checked = 0
    query = ('SELECT r.kind,r.data,i.inference,i.populated,i.signature,t.thresholds,m.format,m.data '
             'FROM results AS r LEFT JOIN result_index AS i USING(scope,scene,model,kind) '
             'LEFT JOIN result_thresholds AS t USING(scope,scene,model,kind) '
             'LEFT JOIN result_metadata AS m USING(scope,scene,model,kind)')
    for kind, blob, inference, populated, signature, threshold, metadata_format, metadata in db.execute(query):
        checked += 1
        tick(progress)
        try:
            segments = decoded_payload(blob, list)
            if not valid_result_segments(segments, kind):
                raise ValueError("Invalid result segment.")
            from stash_ai.storage.archive import validate_active
            validate_active(segments)
            from stash_ai.evidence.formats import validate_current
            for segment in segments:
                validate_current(segment)
            if any(
                    {'scene_threshold_override', 'marker_threshold_override', 'refinement'} & segment.keys()
                    for segment in segments):
                raise ValueError('Noncanonical result segment.')
            if metadata_format is not None:
                from stash_ai.storage.metadata import FORMAT, decode, encode, summarize
                try:
                    if metadata_format != FORMAT:
                        raise ValueError('Unsupported result metadata format.')
                    if encode(decode(metadata)) != encode(summarize(segments)):
                        errors['result_metadata_mismatch'] += 1
                except (TypeError, ValueError):
                    errors['result_metadata_invalid'] += 1
            if threshold is not None:
                from stash_ai.storage.thresholds import decode, overlay
                try:
                    values = decode(threshold)
                    if len(values) != len(segments):
                        errors['result_threshold_invalid'] += 1
                    overlay(segments, kind, threshold)
                except (TypeError, ValueError):
                    pass  # check_thresholds reports invalid policy separately.
            current = next(
                (segment for segment in reversed(segments)
                 if segment["meta"].get("origin") == "inference"),
                None,
            )
            expected_inference = int(current is not None)
            expected_populated = int(any(segment["items"] for segment in segments))
            if current is not None:
                if indexed_signature is None:
                    from stash_ai.evidence.thresholds import indexed_signature as signature_for
                    indexed_signature = signature_for
                expected_signature = indexed_signature(current)
            else:
                expected_signature = None
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError, zlib.error):
            errors["result_payload_invalid"] += 1
            continue

        if inference is None and "result_index" in tables:
            errors["result_missing_index"] += 1
        elif (inference not in (0, 1) or populated not in (0, 1)
              or inference != expected_inference or populated != expected_populated
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
    from stash_ai.storage.thresholds import decode
    for data, in db.execute('SELECT thresholds FROM result_thresholds'):
        try:
            decode(data)
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
