"""Validate state payloads and cleanup/recovery relationships."""
from __future__ import annotations

import json
import zlib

from ..common import KNOWN_KINDS, column_names, count_rows
from ..payloads import valid_frames
from .common import decoded_payload, tick


def check_state_payloads(db, tables, errors, progress=None):
    specs = (
        ("inference_identities", "data", dict),
        ("inference_archive", "data", dict),
        ("snapshots", "published", dict),
        ("snapshots", "pending", dict),
        ("cleanup_state", "data", dict),
        ("checkpoint_frames", "data", list),
        ("checkpoint_analysis", "data", dict),
        ("recovery_publications", "data", dict),
    )
    from .publications import validate_snapshot, proof_checker
    missing_proof = proof_checker(db)
    checked = 0
    for table, column, expected in specs:
        if table not in tables or column not in column_names(db, table):
            continue
        details = (',signature,kind,scope,scene' if table == 'inference_archive'
                   else ',NULL,kind,scope,scene' if table in ('snapshots', 'recovery_publications')
                   else ',NULL,NULL,NULL,NULL')
        for blob, signature, kind, scope, scene in db.execute(f"SELECT {column}{details} FROM {table} WHERE {column} IS NOT NULL"):
            checked += 1
            tick(progress)
            try:
                if table in ('snapshots', 'recovery_publications'):
                    from stash_ai.storage.snapshots import decode
                    payload = decode(blob)
                    validate_snapshot(payload, kind)
                    if missing_proof(payload, scope, scene, kind):
                        errors['snapshot_proof_missing'] += 1
                else:
                    payload = decoded_payload(blob, expected)
                if table == 'inference_identities':
                    from stash_ai.inference.provenance import predictive
                    required = {'format', 'source', 'repo', 'revision', 'files', 'vocabulary',
                                'batch_size', 'scene_threshold', 'marker_threshold', 'aggregation', 'top_k', 'retention'}
                    if not required <= payload.keys() or not isinstance(payload['retention'], dict):
                        raise ValueError('Incomplete inference identity.')
                    predictive(payload)
                if table == 'inference_archive':
                    from stash_ai.db.payloads import valid_result
                    from stash_ai.evidence.formats import validate_current
                    if not valid_result(payload, kind):
                        raise ValueError('Invalid archived inference proof.')
                    validate_current(payload)
                    from stash_ai.evidence.identity import marker_segment_signature
                    if marker_segment_signature(payload) != signature:
                        raise ValueError('Archived proof signature mismatch.')
                if table == 'checkpoint_analysis':
                    from stash_ai.evidence.temporal.statistics import validate_checkpoint
                    validate_checkpoint(payload)
                if table == "checkpoint_frames" and not valid_frames(payload, checkpoint=True):
                    raise ValueError("Invalid checkpoint frame evidence.")
            except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError, zlib.error):
                errors["state_payload_invalid"] += 1
    return checked


def check_cleanup_state(db, tables, errors, progress=None):
    generations = {}
    if "cleanup_state" in tables:
        for scope, generation, blob in db.execute("SELECT scope,generation,data FROM cleanup_state"):
            tick(progress)
            generations[scope] = generation
            invalid = type(generation) is not int or generation < 0
            try:
                masks = decoded_payload(blob, dict)
                for values in masks.values():
                    if not isinstance(values, dict):
                        invalid = True
                        continue
                    for revision in values.values():
                        if type(revision) is not int or revision <= 0 or revision > generation:
                            invalid = True
            except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError, zlib.error):
                # Counted by the generic BLOB check as well; this code reports
                # the higher-level cleanup invariant, not a second payload error.
                invalid = True
            errors["cleanup_state_invalid"] += int(invalid)

    if "recovery_cleanups" in tables:
        for scope, generation, kinds in db.execute(
                "SELECT scope,generation,kinds FROM recovery_cleanups"):
            tick(progress)
            invalid = type(generation) is not int or generation <= 0
            try:
                decoded = json.loads(kinds)
                invalid |= (
                    not isinstance(decoded, list)
                    or not decoded
                    or any(kind not in KNOWN_KINDS for kind in decoded)
                    or len(decoded) != len(set(decoded))
                )
            except (TypeError, json.JSONDecodeError):
                invalid = True
            current = generations.get(scope)
            if current is None or generation > current:
                invalid = True
            errors["recovery_cleanup_invalid"] += int(invalid)


def check_state_relations(db, tables, warnings):
    if "snapshots" in tables:
        if "recovery_publications" in tables:
            warnings["empty_snapshot_without_recovery"] += db.execute("""
                SELECT COUNT(*) FROM snapshots AS s
                WHERE s.published IS NULL AND s.pending IS NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM recovery_publications AS r
                    WHERE r.scope=s.scope AND r.scene=s.scene AND r.kind=s.kind
                  )
            """).fetchone()[0]
        else:
            warnings["empty_snapshot_without_recovery"] += db.execute(
                "SELECT COUNT(*) FROM snapshots WHERE published IS NULL AND pending IS NULL"
            ).fetchone()[0]

    if "recovery_publications" in tables:
        if "snapshots" in tables:
            warnings["recovery_publication_without_snapshot"] += db.execute("""
                SELECT COUNT(*) FROM recovery_publications AS r
                WHERE NOT EXISTS (
                    SELECT 1 FROM snapshots AS s
                    WHERE s.scope=r.scope AND s.scene=r.scene AND s.kind=r.kind
                )
            """).fetchone()[0]
        else:
            warnings["recovery_publication_without_snapshot"] += count_rows(
                db, tables, "recovery_publications")
