"""Replay retained VTT observations in private RAM with explicit local truth.

No model loading, media reads, writer lock, schema migration or Stash calls.
Every configuration starts cold; batched runs see only previously admitted
scenes. Unlabelled observations still participate as identity competitors.
"""
from collections import Counter
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from .clustering import matching_policy
from .config import digest, load, plain
from .database import SCHEMA, TABLES, FaceDatabase, encoded
from .decision_evidence import recorded, explanation
from .evaluation import compare_identity_runs
from .fragmentation_report import _open_readonly
from .models import namespace
from .rebuild import rebuild_selected
from .scanning import scan_identity


def _ram(config):
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA temp_store=MEMORY')
    page = db.execute('PRAGMA page_size').fetchone()[0]
    db.execute(f"PRAGMA max_page_count={max(1, config.data['execution']['rebuild_memory_bytes']//page)}")
    db.executescript(SCHEMA)
    return db


def _selection(db, config, scenes, scope):
    ns = namespace(config)
    scopes = [r[0] for r in db.execute('SELECT DISTINCT scope FROM scans WHERE namespace=?', (ns,))]
    if scope is None:
        if len(scopes) != 1:
            raise ValueError('Evaluation requires one explicit scope for this recognizer.')
        scope = scopes[0]
    wanted = set(scenes or ())
    if any(not isinstance(s, str) or not s.isascii() or not s.isdecimal() or int(s) < 1 for s in wanted):
        raise ValueError('Evaluation scene IDs must be positive decimal strings.')
    scans = [dict(r) for r in db.execute('SELECT s.* FROM scans s WHERE scope=? AND namespace=? '
        'AND NOT EXISTS (SELECT 1 FROM scans n WHERE n.scope=s.scope AND n.namespace=s.namespace '
        'AND n.scene=s.scene AND n.rowid>s.rowid) ORDER BY CAST(s.scene AS INTEGER),s.scene', (scope, ns))
        if not wanted or r['scene'] in wanted]
    if not scans or wanted and {s['scene'] for s in scans} != wanted:
        raise ValueError('Evaluation scenes are missing from the selected scope/namespace.')
    if any(s['status'] != 'complete' or s['policy_sha'] != scan_identity(config, ns) for s in scans):
        raise ValueError('Evaluation requires latest complete VTT scans matching the detection policy; rescan first.')
    return scope, scans


def _copy_scene(source, target, scan):
    target.execute('INSERT INTO scans VALUES (' + ','.join('?' for _ in scan) + ')', tuple(scan.values()))
    rows = source.execute('SELECT id,scan,cue,seconds,source,rect,geometry,quality,rejected,dimensions,embedding,NULL '
                          'FROM observations WHERE scan=? ORDER BY cue,id', (scan['id'],))
    target.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    target.commit()


def evaluate(config, labels, *, scenes=None, scope=None, variants=None, batch_sizes=(0, 1)):
    """Labels map retained observation IDs to human person labels, not predictions."""
    if not isinstance(labels, dict) or not labels or any(not isinstance(k, str) or not k or
            not isinstance(v, str) or not v.strip() for k, v in labels.items()):
        raise ValueError('Evaluation requires a nonempty observation-ID to truth-label object.')
    if not batch_sizes or len(set(batch_sizes)) != len(batch_sizes) or any(type(n) is not int or n < 0 for n in batch_sizes):
        raise ValueError('Batch sizes must be distinct nonnegative integers; 0 means joint replay.')
    configurations = {'baseline': config}
    for name, variant in (variants or {}).items():
        if not isinstance(name, str) or not name or name in configurations:
            raise ValueError('Evaluation configuration names must be unique and nonempty.')
        if namespace(variant) != namespace(config) or scan_identity(variant, namespace(config)) != scan_identity(config, namespace(config)):
            raise ValueError('Evaluation variants must preserve the model, detection and VTT scan contracts.')
        configurations[name] = variant
    predictions, trials = {}, {}
    with closing(_open_readonly(config.database)) as source:
        tables = {r[0] for r in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        FaceDatabase._validate(SimpleNamespace(connection=source), SCHEMA, TABLES, tables)
        scope, scans = _selection(source, config, scenes, scope)
        installation = FaceDatabase.installation.fget(SimpleNamespace(connection=source))
        wanted = encoded([s['id'] for s in scans])
        known = {r['id']: dict(observation=r['id'], scene=r['scene'], truth=labels[r['id']])
                 for r in source.execute('SELECT o.id,s.scene FROM observations o JOIN scans s ON s.id=o.scan '
                    'WHERE o.scan IN (SELECT value FROM json_each(?))', (wanted,)) if r['id'] in labels}
        if set(known) != set(labels):
            raise ValueError('Every truth label must address an observation in the selected latest scans.')
        for name, variant in configurations.items():
            data = plain(variant.data)
            # Explicit evaluation never writes durable replay checkpoints.
            data['execution']['rebuild_checkpoint_scenes'] = 0
            trial_config = replace(variant, data=data)
            for size in batch_sizes:
                key = name + ('/joint' if size == 0 else f'/batch-{size}')
                with closing(_ram(trial_config)) as target:
                    target.execute('INSERT INTO metadata VALUES (?,?)', ('installation', installation))
                    target.commit()
                    state = SimpleNamespace(connection=target, installation=installation)
                    width = size or len(scans)
                    for offset in range(0, len(scans), width):
                        batch = scans[offset:offset+width]
                        for scan in batch:
                            _copy_scene(source, target, scan)
                        rebuild_selected(state, [dict(scene=s['scene'], scan=s['id']) for s in batch], trial_config)
                    members = dict(target.execute('SELECT observation,person FROM members'))
                    predictions[key] = {identifier: members.get(identifier) for identifier in labels}
                    decisions, reasons = {}, Counter()
                    for row in target.execute('SELECT c.* FROM candidates c WHERE c.policy=? ORDER BY c.rowid',
                                              (matching_policy(trial_config),)):
                        payload = json.loads(row['payload'])
                        reasons[payload.get('reason', 'unknown')] += 1
                        if payload.get('observation') in labels:
                            decisions[payload['observation']] = dict(status=row['status'], identity=row['person'],
                                evidence=recorded(payload), explanation=explanation(payload))
                    for identifier in labels.keys() - decisions.keys():
                        rejected = target.execute('SELECT rejected FROM observations WHERE id=?', (identifier,)).fetchone()[0]
                        decisions[identifier] = dict(status='rejected' if rejected else 'withheld', identity=None,
                            evidence=dict(observation=identifier, reason=rejected or 'no_assignment_decision'),
                            explanation='No accepted production decision for this retained observation.')
                    trials[key] = dict(matching_policy=matching_policy(trial_config),
                        config=plain(trial_config.data['recognition']), clustering=plain(trial_config.data['clustering']),
                        batch_size=size, reason_counts=dict(sorted(reasons.items())), decisions=decisions)
    return dict(format=1, kind='recognition_evaluation', source_read_only=True, source='retained_vtt_embeddings',
        starting_state='cold_selected_scenes_only', scope=scope, selection=[s['scene'] for s in scans],
        labels_sha256=digest(labels), labelled_observations=len(labels),
        metrics=compare_identity_runs(list(known.values()), predictions), trials=trials,
        limitations=['Quality metrics describe only the explicitly labelled observations.',
                     'Unlabelled selected observations still compete; external historical identities are excluded.',
                     'Batches admit scenes in numeric ID order; this is not a media timeline.'])


def execute(args, config):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate evaluation JSON key: ' + key)
            result[key] = value
        return result
    labels = json.loads(args.labels.read_text(encoding='utf-8-sig'), object_pairs_hook=unique)
    variants = {}
    for path in args.compare_config or ():
        name = path.stem
        if name == 'baseline' or name in variants:
            raise ValueError('Comparison YAML filenames must have distinct stems other than baseline.')
        variants[name] = load(path)
    result = evaluate(config, labels, scenes=args.scene_id, scope=args.scope, variants=variants,
                      batch_sizes=tuple(args.batch_size if args.batch_size is not None else (0, 1)))
    output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output is None:
        print(output, end='')
    else:
        with args.output.expanduser().open('x', encoding='utf-8') as file:
            file.write(output)
        print(f'Recognition evaluation written: {args.output}')
    return 0
