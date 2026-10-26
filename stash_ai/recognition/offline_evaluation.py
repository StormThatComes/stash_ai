"""Replay retained VTT observations in private RAM with explicit local truth.

No model loading, media reads, writer lock, schema migration or Stash calls.
Trials explicitly choose cold or retained external history. Batched admission
excludes future selected scenes. Unlabelled observations remain competitors.
"""
from contextlib import closing
from dataclasses import replace
import json
import sqlite3
import time
from itertools import product
from types import SimpleNamespace

from .clustering import matching_policy
from .config import digest, load, plain
from .database import SCHEMA, TABLES, FaceDatabase, encoded
from .evaluation_scenarios import ordered, copy_history, snapshot
from .person_signal import coverage, same_contract
from .evaluation import compare_identity_runs
from .fragmentation_report import _open_readonly
from .models import namespace
from .rebuild import rebuild_selected
from .scanning import scan_identity


def _ram(config):
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('PRAGMA temp_store=MEMORY')
    page = db.execute('PRAGMA page_size').fetchone()[0]
    db.execute(f"PRAGMA max_page_count={max(1, config.data['execution']['rebuild_memory_bytes']//page)}")
    db.executescript(SCHEMA)
    return db


def _selection(db, config, scenes, scope, *, allow_partial=False, diagnostics=None):
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
    from .replay_inputs import classify, ReplayInputError
    scans, detail = classify(scans, scan_identity(config, ns))
    if diagnostics is not None:
        diagnostics.update(detail)
    if not allow_partial and not detail['complete_scope']:
        raise ReplayInputError(detail)
    return scope, scans


def _copy_scene(source, target, scan):
    target.execute('INSERT INTO scans VALUES (' + ','.join('?' for _ in scan) + ')', tuple(scan.values()))
    rows = source.execute('SELECT id,scan,cue,seconds,source,rect,geometry,quality,rejected,dimensions,embedding,NULL '
                          'FROM observations WHERE scan=? ORDER BY cue,id', (scan['id'],))
    target.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    from .person_signal import copy_evidence
    copy_evidence(source,target,scan['id'])
    from .pixel_storage import copy_rows
    copy_rows(source,target,source.execute('SELECT quality FROM observations WHERE scan=?',(scan['id'],)))
    target.commit()


def evaluate(config, labels, *, scenes=None, scope=None, variants=None, batch_sizes=(0, 1),
             scene_orders=('numeric',), shuffle_seed=0, history_modes=('cold',), repeat_runs=1,
             retune_from_baseline=False):
    """Labels map retained observation IDs to human person labels, not predictions."""
    if not isinstance(labels, dict) or not labels or any(not isinstance(k, str) or not k or
            not isinstance(v, str) or not v.strip() for k, v in labels.items()):
        raise ValueError('Evaluation requires a nonempty observation-ID to truth-label object.')
    if not batch_sizes or len(set(batch_sizes)) != len(batch_sizes) or any(type(n) is not int or n < 0 for n in batch_sizes):
        raise ValueError('Batch sizes must be distinct nonnegative integers; 0 means joint replay.')
    if (not scene_orders or len(set(scene_orders)) != len(scene_orders) or
            set(scene_orders) - {'numeric', 'reverse', 'shuffle'}):
        raise ValueError('Evaluation scene orders must be distinct numeric/reverse/shuffle values.')
    if (not history_modes or len(set(history_modes)) != len(history_modes) or
            set(history_modes) - {'cold', 'retained'}):
        raise ValueError('Evaluation history modes must be distinct cold/retained values.')
    if type(repeat_runs) is not int or repeat_runs < 1 or type(shuffle_seed) is not int:
        raise ValueError('Evaluation requires positive repeat_runs and an integer shuffle_seed.')
    if type(retune_from_baseline) is not bool:
        raise ValueError('retune_from_baseline must be boolean.')
    configurations = {'baseline': config}
    for name, variant in (variants or {}).items():
        if not isinstance(name, str) or not name or name in configurations:
            raise ValueError('Evaluation configuration names must be unique and nonempty.')
        if namespace(variant) != namespace(config) or scan_identity(variant, namespace(config)) != scan_identity(config, namespace(config)) or not same_contract(config,variant):
            raise ValueError('Evaluation variants must preserve the model, person, detection and VTT scan contracts.')
        configurations[name] = variant
    predictions, trials = {}, {}
    with closing(_open_readonly(config.database)) as source:
        tables = {r[0] for r in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        from .database import SCHEMA_V5, TABLES_V5, SCHEMA_V6, TABLES_V6
        schema, expected = (SCHEMA, TABLES) if 'evidence_thumbnails' in tables else (SCHEMA_V6,TABLES_V6) if 'person_evidence' in tables else (SCHEMA_V5,TABLES_V5)
        FaceDatabase._validate(SimpleNamespace(connection=source), schema, expected, tables)
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
            for size, order, history_mode in product(batch_sizes, scene_orders, history_modes):
                trial_scans = ordered(scans, order, shuffle_seed)
                base_key = name + ('/joint' if size == 0 else f'/batch-{size}')
                if scene_orders != ('numeric',) or order != 'numeric':
                    base_key += '/' + order
                if history_modes != ('cold',) or history_mode != 'cold':
                    base_key += '/' + history_mode
                retuning = retune_from_baseline and name != 'baseline'
                if retuning:
                    base_key += '/retuned'
                with closing(_ram(trial_config)) as target:
                    target.execute('INSERT INTO metadata VALUES (?,?)', ('installation', installation))
                    target.commit()
                    history = (copy_history(source, target, scans, scope, namespace(config))
                               if history_mode == 'retained' else dict(scenes=0, observations=0, people=0))
                    state = SimpleNamespace(connection=target, installation=installation)
                    width = size or len(scans)
                    loaded = set()
                    def replay(settings):
                        for offset in range(0, len(trial_scans), width):
                            batch = trial_scans[offset:offset+width]
                            for scan in batch:
                                if scan['id'] not in loaded:
                                    _copy_scene(source, target, scan)
                                    loaded.add(scan['id'])
                            rebuild_selected(state, [dict(scene=s['scene'], scan=s['id']) for s in batch], settings)
                    if retuning:
                        baseline_data = plain(config.data)
                        baseline_data['execution']['rebuild_checkpoint_scenes'] = 0
                        replay(replace(config, data=baseline_data))
                    for run in range(1, repeat_runs + 1):
                        key = base_key + (f'/run-{run}' if repeat_runs != 1 else '')
                        started = time.perf_counter()
                        replay(trial_config)
                        elapsed = time.perf_counter() - started
                        predictions[key], decisions, reasons = snapshot(target, trial_config, labels)
                        trials[key] = dict(hybrid=coverage(target,trial_config,loaded),matching_policy=matching_policy(trial_config),
                            config=plain(trial_config.data['recognition']), clustering=plain(trial_config.data['clustering']),
                            batch_size=size, scene_order=order, selection=[s['scene'] for s in trial_scans],
                            shuffle_seed=shuffle_seed if order == 'shuffle' else None,
                            history_mode=history_mode, history=history, run=run,
                            retuned_from_baseline=retuning, elapsed_seconds=elapsed,
                            reason_counts=reasons, decisions=decisions)
    return dict(format=2, kind='recognition_evaluation', source_read_only=True, source='retained_vtt_embeddings',
        starting_state='per_trial_history_mode', scope=scope, selection=[s['scene'] for s in scans],
        labels_sha256=digest(labels), labelled_observations=len(labels),
        metrics=compare_identity_runs(list(known.values()), predictions), trials=trials,
        limitations=['Quality metrics describe only the explicitly labelled observations.',
                     'Unlabelled observations compete; retained trials include frozen external historical evidence.',
                     'Import order is explicit per trial and is not a media timeline.',
                     'History is a snapshot of existing automatic decisions, not human ground truth.',
                     'Elapsed time measures replay only; it excludes model inference, media and history copying.'])


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
                      batch_sizes=tuple(args.batch_size if args.batch_size is not None else (0, 1)),
                      scene_orders=tuple(args.scene_order or ('numeric',)), shuffle_seed=args.shuffle_seed,
                      history_modes=tuple(args.history or ('cold',)), repeat_runs=args.repeat_runs,
                      retune_from_baseline=args.retune_from_baseline)
    output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output is None:
        print(output, end='')
    else:
        with args.output.expanduser().open('x', encoding='utf-8') as file:
            file.write(output)
        print(f'Recognition evaluation written: {args.output}')
    return 0
