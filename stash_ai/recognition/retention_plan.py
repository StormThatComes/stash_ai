"""Conservative, disk-backed retention plans. No changes to main tables."""
from itertools import groupby

from .database import encoded
from .profiling import pipeline_timing, pipeline_count
from .scan_heads import predicate
from .progress import stage, tracking, work


TEMP_TABLES = ('retention_scans', 'retention_success', 'retention_targets', 'retention_pins',
               'retention_actions', 'retention_contracts', 'retention_thumbnails')
REFERENCE_BATCH_ROWS = 256


def discard(db):
    for name in reversed(TEMP_TABLES):
        db.execute('DROP TABLE IF EXISTS temp.' + name)


def prepare(db, config, *, scope=None, scenes=None, writable=False):
    """Caller holds one snapshot/transaction through planning and application."""
    from .models import namespace
    from .scanning import scan_identity
    with stage('Initialize retention plan'):
        discard(db)
        db.execute('CREATE TEMP TABLE retention_scans(id TEXT PRIMARY KEY, action TEXT, reason TEXT)')
        db.execute('CREATE TEMP TABLE retention_success(scan TEXT PRIMARY KEY)')
        db.execute('CREATE TEMP TABLE retention_targets(id TEXT PRIMARY KEY, scan TEXT NOT NULL)')
        db.execute('CREATE TEMP TABLE retention_pins(id TEXT PRIMARY KEY, image INTEGER NOT NULL)')
        db.execute('CREATE TEMP TABLE retention_actions(table_name TEXT, id TEXT, contract TEXT, action TEXT, '
                   'PRIMARY KEY(table_name,id,contract))')
        db.execute('CREATE TEMP TABLE retention_contracts(scan TEXT, kind TEXT, contract TEXT, '
                   'PRIMARY KEY(scan,kind,contract))')
        db.execute('CREATE TEMP TABLE retention_thumbnails(id TEXT PRIMARY KEY)')
    model = namespace(config)
    policy = scan_identity(config, model)
    settings = config.data['storage']['retention']
    with stage('Read successful replacement runs'), pipeline_timing('retention_success_runs'):
        # Reference indexing is deferred until a deletion candidate exists.
        from .retention_index import refresh
        indexed = refresh(db, writable, success_only=True)
        if not indexed:db.execute("INSERT OR IGNORE INTO retention_success SELECT json_extract(j.value,'$.scan') "
                   "FROM progress p JOIN runs r ON r.id=p.run, json_each(p.data,'$.selection') j "
                   "WHERE r.status='success' AND p.phase='rebuild' AND json_extract(j.value,'$.scan') IS NOT NULL")
    with stage('Select retained scans'), pipeline_timing('retention_scan_selection'):
        rows = db.execute('SELECT s.*,(' + predicate() + ') AS active FROM scans s WHERE namespace=? '
                          'AND (? IS NULL OR scope=?) AND (? IS NULL OR scene IN (SELECT value FROM json_each(?))) '
                          'ORDER BY scope,scene,rowid DESC',
                          (model, scope, scope, None if scenes is None else encoded(scenes), encoded(scenes)))
        with tracking(groupby(rows, lambda r: (r['scope'], r['scene'])),
                      'Inspect retention scenes', unit='scene') as groups:
            for _, group in groups:
                scans = list(group)
                active = next((r for r in scans if r['active']), None)
                blocked = None
                if active is None or active['status'] != 'complete' or active['cursor'] != active['total']:
                    blocked = 'active_scan_incomplete'
                elif active['policy_sha'] != policy:
                    blocked = 'configuration_differs_from_active_scan'
                elif not db.execute(("SELECT 1 FROM retention_verified v JOIN runs r ON r.id=v.run "
                    "JOIN progress p ON p.run=v.run AND p.phase='rebuild' WHERE v.scan=? AND r.status='success' LIMIT 1"
                    if indexed else 'SELECT 1 FROM retention_success WHERE scan=?'), (active['id'],)).fetchone():
                    blocked = 'no_successful_replacement_run'
                elif db.execute("SELECT 1 FROM publications WHERE scope=? AND scene=? "
                                "AND status NOT IN ('confirmed','observed') LIMIT 1",
                                (active['scope'], active['scene'])).fetchone():
                    blocked = 'unresolved_publication'
                # A reconciliation queue may have no publication intent yet.
                if not blocked:
                    from .publication_queue import key
                    if db.execute('SELECT 1 FROM metadata WHERE key=?',
                                  (key(active['scope'], model, active['scene']),)).fetchone():
                        blocked = 'pending_scene_reconciliation'
                previous = settings['keep_previous_scans']
                for index, scan in enumerate(scans):
                    reason = blocked
                    if not reason:
                        if scan['active']:
                            reason = 'active_scan'
                        elif scan['status'] != 'complete' or scan['cursor'] != scan['total']:
                            reason = 'incomplete_scan'
                        elif previous:
                            reason = 'previous_scan'
                            previous -= 1
                        elif index == 0:
                            # The remembered-head marker refers to the newest created ID.
                            reason = 'head_creation_marker'
                    db.execute('INSERT INTO retention_scans VALUES (?,?,?)',
                               (scan['id'], 'keep' if reason else 'obsolete', reason))
    with stage('Select obsolete person contracts'), pipeline_timing('retention_contract_selection'):
        _select_contracts(db, config)
    with pipeline_timing('retention_target_selection'):
        _targets(db)
    if db.execute('SELECT 1 FROM retention_targets LIMIT 1').fetchone():
        with stage('Protect identity and audit references'), pipeline_timing('retention_reference_protection'):
            from .retention_index import refresh
            indexed = refresh(db, writable)
            _pins(db, indexed)
    else:
        pipeline_count('retention_empty_targets')
        if not db.execute("SELECT 1 FROM retention_scans WHERE action='obsolete' LIMIT 1").fetchone():
            return

    with stage('Plan obsolete scans and crops'):
        # Keep immutable numerical/geometry evidence that still owns identity roots.
        db.execute("UPDATE retention_scans SET action='archive',reason='identity_or_audit_reference' "
                   "WHERE action='obsolete' AND (EXISTS (SELECT 1 FROM observations o JOIN retention_pins p ON p.id=o.id "
                   "WHERE o.scan=retention_scans.id) OR EXISTS (SELECT 1 FROM body_observations b "
                   "JOIN retention_pins p ON p.id=b.id WHERE b.scan=retention_scans.id))")
        db.execute("UPDATE retention_scans SET action='delete' WHERE action='obsolete'")
        for table, address in (('observations', 'id'), ('body_observations', 'id')):
            db.execute(f"INSERT INTO retention_actions SELECT '{table}',o.{address},'', 'delete' FROM {table} o "
                       "JOIN retention_scans s ON s.id=o.scan WHERE s.action='delete'")
            if settings['prune_obsolete_crops']:
                db.execute(f"INSERT INTO retention_actions SELECT '{table}',o.{address},'', 'clear_crop' FROM {table} o "
                           "JOIN retention_scans s ON s.id=o.scan WHERE s.action='archive' AND o.crop IS NOT NULL "
                           "AND NOT EXISTS (SELECT 1 FROM retention_pins p WHERE p.id=o.id AND p.image>=1)")
        db.execute("INSERT INTO retention_actions SELECT 'person_evidence',p.observation,p.contract,'delete' "
                   "FROM person_evidence p JOIN observations o ON o.id=p.observation "
                   "JOIN retention_scans s ON s.id=o.scan WHERE s.action='delete'")
        if settings['prune_obsolete_crops']:
            db.execute("INSERT INTO retention_actions SELECT 'person_evidence',p.observation,p.contract,'clear_crop' "
                       "FROM person_evidence p JOIN observations o ON o.id=p.observation "
                       "JOIN retention_scans s ON s.id=o.scan WHERE s.action='archive' AND p.crop IS NOT NULL "
                       "AND NOT EXISTS (SELECT 1 FROM retention_pins pin WHERE pin.id=o.id AND pin.image>=1)")
    with stage('Plan obsolete person contracts'), pipeline_timing('retention_contract_actions'):
        _contracts(db, config)


def _targets(db):
    for table in ('observations', 'body_observations'):
        db.execute(f'INSERT OR IGNORE INTO retention_targets SELECT o.id,o.scan FROM {table} o '
                   "JOIN retention_scans s ON s.id=o.scan WHERE s.action='obsolete'")
    db.execute("INSERT OR IGNORE INTO retention_targets SELECT b.id,b.scan FROM body_observations b "
               "JOIN retention_contracts c ON c.scan=b.scan AND c.kind='body' AND c.contract=b.contract")
    db.execute("INSERT OR IGNORE INTO retention_targets SELECT o.id,o.scan FROM observations o "
               "JOIN person_evidence p ON p.observation=o.id JOIN retention_contracts c "
               "ON c.scan=o.scan AND c.kind='person' AND c.contract=p.contract")
    pipeline_count('retention_target_rows', db.execute('SELECT COUNT(*) FROM retention_targets').fetchone()[0])


def _pins(db,indexed=False):
    def pin(select, image=0):
        db.execute('INSERT INTO retention_pins SELECT id,? FROM (' + select + ') '
                   'WHERE id IN (SELECT id FROM retention_targets) '
                   'ON CONFLICT(id) DO UPDATE SET image=MAX(image,excluded.image)', (image,))
    for table in ('members', 'body_members'):
        pin(f'SELECT observation AS id FROM {table}', 1)
    for table in ('identity_origins', 'body_origins', 'merge_support'):
        pin(f'SELECT observation AS id FROM {table}')
    # Cross-scan face associations cannot be removed with their source scan.
    pin('SELECT b.face AS id FROM body_observations b JOIN observations o ON o.id=b.face WHERE b.scan<>o.scan')
    # JSON evidence also contains addresses without SQL foreign keys. Preserve
    # referenced rows, including nested merge witnesses and historical portraits.
    for table, column, condition, image in (
        ('identity_merges', 'evidence', '1', 0),
        ('body_members', 'proof', '1', 0),
        ('portraits', 'metadata', '1', 2),
        ('publications', 'payload', '1', 2),
        ('publications', 'precondition', '1', 2),
        ('candidates', 'payload', "(t.scan NOT IN (SELECT id FROM retention_scans WHERE action='obsolete') OR e.scan<>t.scan)", 0),
    ):
        with stage(f'Protect references: {table}.{column}'):
            if indexed:
                from .retention_index import pins
                pins(db,table,column,condition,image)
                # Non-UUID legacy addresses may have appeared after their source
                # document was indexed. Always inspect those against live JSON.
                from .retention_index import uuid_address
                legacy = 'NOT (' + uuid_address('e.id') + ')'
                if db.execute('SELECT 1 FROM retention_targets e WHERE '+legacy+' LIMIT 1').fetchone():
                    _pin_json(db,table,column,'('+condition+') AND '+legacy,image)
            else:
                _pin_json(db, table, column, condition, image)


def _pin_json(db, table, column, condition, image):
    """Visit each source document once, with bounded batches and indexed pins.

    These sources can reference evidence outside their own scope/scene. Keep
    scanning all source rows; only the target lookup is selection-scoped.
    """
    with work(f'Inspect references: {table}.{column}',
              total=lambda: db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0],
              unit='record') as progress:
        cursor = None
        while True:
            where = '' if cursor is None else 'WHERE rowid>?'
            parameters = () if cursor is None else (cursor,)
            low, high, count = db.execute(
                f'SELECT MIN(rowid),MAX(rowid),COUNT(*) FROM '
                f'(SELECT rowid FROM {table} {where} ORDER BY rowid LIMIT ?)',
                (*parameters, REFERENCE_BATCH_ROWS)).fetchone()
            if not count:
                break
            bounds = (low, high)
            if db.execute(f'SELECT 1 FROM {table} WHERE rowid BETWEEN ? AND ? '
                          f'AND NOT json_valid({column}) LIMIT 1', bounds).fetchone():
                raise ValueError(f'Retention cannot inspect malformed {table}.{column}; no data was pruned.')
            # CROSS JOIN fixes the loop order: source -> JSON leaves -> target
            # PK lookup. The former outer IN filter let SQLite put targets first
            # and parse every source JSON again for every possible target.
            db.execute('INSERT INTO retention_pins SELECT e.id,? '
                       f'FROM {table} t CROSS JOIN json_tree(t.{column}) j '
                       'CROSS JOIN retention_targets e '
                       "WHERE t.rowid BETWEEN ? AND ? AND j.type='text' AND e.id=j.value "
                       f'AND ({condition}) '
                       'ON CONFLICT(id) DO UPDATE SET image=MAX(image,excluded.image)',
                       (image, *bounds))
            cursor = high
            # Flush before another potentially slow SQLite batch, rather than
            # leaving the last visible count behind the work already completed.
            progress.advance(count, force=True)


def _select_contracts(db, config):
    from .person_signal import contract as person_contract
    from .body_observations import contract as body_contract, enabled
    if not config.data['sapiensid']['enabled']:
        return
    settings = config.data['storage']['retention']
    # Only prune variants in scans whose current face policy succeeded. Leave
    # archived/incomplete/blocked scans' numerical evidence untouched.
    scans = db.execute("SELECT s.id,s.total FROM scans s JOIN retention_scans r ON r.id=s.id "
                       "WHERE r.reason IN ('active_scan','previous_scan')")
    with tracking(scans, 'Inspect person contracts', unit='scan') as items:
        for scan, total in items:
            for kind, expected in (('person', person_contract(config)), ('body', body_contract(config))):
                if kind == 'body':
                    if not enabled(config):
                        continue
                    complete = db.execute('SELECT COUNT(*),MIN(cue),MAX(cue) FROM body_cues '
                                          'WHERE scan=? AND contract=?', (scan, expected)).fetchone()
                    if not total or tuple(complete) != (total, 0, total - 1):
                        continue
                    variants = [r[0] for r in db.execute('SELECT contract FROM body_cues WHERE scan=? '
                                'GROUP BY contract ORDER BY MAX(rowid) DESC', (scan,))]
                else:
                    if not db.execute('SELECT 1 FROM person_evidence p JOIN observations o ON o.id=p.observation '
                                      'WHERE o.scan=? AND p.contract=? LIMIT 1', (scan, expected)).fetchone():
                        continue
                    if db.execute('SELECT 1 FROM observations o WHERE o.scan=? AND o.rejected IS NULL AND NOT EXISTS '
                                  '(SELECT 1 FROM person_evidence p WHERE p.observation=o.id AND p.contract=?) LIMIT 1',
                                  (scan, expected)).fetchone():
                        continue
                    variants = [r[0] for r in db.execute('SELECT p.contract FROM person_evidence p '
                                'JOIN observations o ON o.id=p.observation WHERE o.scan=? '
                                'GROUP BY p.contract ORDER BY MAX(p.rowid) DESC', (scan,))]
                previous = settings['keep_previous_person_contracts']
                for variant in variants:
                    if variant == expected:
                        continue
                    if kind == 'body':
                        coverage = db.execute('SELECT COUNT(*),MIN(cue),MAX(cue) FROM body_cues '
                                              'WHERE scan=? AND contract=?', (scan, variant)).fetchone()
                        if tuple(coverage) != (total, 0, total - 1):
                            continue  # resumable partial variants do not consume rollback slots
                    elif db.execute('SELECT 1 FROM observations o WHERE o.scan=? AND o.rejected IS NULL AND NOT EXISTS '
                                    '(SELECT 1 FROM person_evidence p WHERE p.observation=o.id AND p.contract=?) LIMIT 1',
                                    (scan, variant)).fetchone():
                        continue
                    if previous:
                        previous -= 1
                        continue
                    db.execute('INSERT INTO retention_contracts VALUES (?,?,?)', (scan, kind, variant))


def _contracts(db, config):
    settings = config.data['storage']['retention']
    for scan, kind, variant in db.execute('SELECT scan,kind,contract FROM retention_contracts'):
        if kind == 'body':
            source = 'body_observations o'
            where = 'o.scan=? AND o.contract=?'
            address, contract, table = 'o.id', "''", 'body_observations'
            # body_cues certifies the complete variant, so never delete
            # just some detections behind a still-complete certificate.
            deletable = not db.execute('SELECT 1 FROM body_observations b JOIN retention_pins p ON p.id=b.id '
                                       'WHERE b.scan=? AND b.contract=? LIMIT 1', (scan, variant)).fetchone()
        else:
            source = 'person_evidence o JOIN observations f ON f.id=o.observation'
            where = 'f.scan=? AND o.contract=?'
            address, contract, table = 'o.observation', 'o.contract', 'person_evidence'
            deletable = not db.execute('SELECT 1 FROM person_evidence p JOIN observations f ON f.id=p.observation '
                                       'JOIN retention_pins pin ON pin.id=p.observation WHERE f.scan=? '
                                       'AND p.contract=? LIMIT 1', (scan, variant)).fetchone()
        if deletable:
            db.execute(f"INSERT OR IGNORE INTO retention_actions SELECT '{table}',{address},{contract},'delete' "
                       f'FROM {source} WHERE {where} AND NOT EXISTS '
                       f'(SELECT 1 FROM retention_pins p WHERE p.id={address})', (scan, variant))
        if settings['prune_obsolete_crops']:
            db.execute(f"INSERT OR IGNORE INTO retention_actions SELECT '{table}',{address},{contract},'clear_crop' "
                       f'FROM {source} WHERE {where} AND o.crop IS NOT NULL AND NOT EXISTS '
                       f'(SELECT 1 FROM retention_pins p WHERE p.id={address} AND p.image>={2 if kind == "person" else 1})', (scan, variant))

def summarize(db):
    from .cold_payloads import length_sql,available
    crop_size=length_sql('o.crop') if available(db) else 'length(o.crop)'
    tables = []
    for table, address in (('observations', 'id'), ('body_observations', 'id'), ('person_evidence', 'observation')):
        join = f'a.id=o.{address}' + (' AND a.contract=o.contract' if table == 'person_evidence' else '')
        columns = [r[1] for r in db.execute(f'PRAGMA table_info({table})')]
        size = '+'.join(f'COALESCE(length(CAST(o.{c} AS BLOB)),0)' for c in columns)
        for action, count, size in db.execute(
                f'SELECT a.action,COUNT(*),SUM(CASE WHEN a.action=\'clear_crop\' THEN {crop_size} ELSE {size} END) '
                f'FROM {table} o JOIN retention_actions a ON {join} WHERE a.table_name=? GROUP BY a.action', (table,)):
            tables.append(dict(table=table, action=action, rows=count, logical_bytes=size or 0))
    return dict(scans=dict(db.execute('SELECT action,COUNT(*) FROM retention_scans GROUP BY action')),
                protected_reasons=dict(db.execute('SELECT reason,COUNT(*) FROM retention_scans '
                                                 'WHERE reason IS NOT NULL GROUP BY reason')),
                obsolete_person_contracts=db.execute('SELECT COUNT(*) FROM retention_contracts').fetchone()[0],
                actions=tables, reclaimable_payload_bytes=sum(t['logical_bytes'] for t in tables))
