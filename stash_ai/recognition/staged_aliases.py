"""Retained-only validation of a component larger than a single scene batch."""
from .config import digest
from .database import encoded
from .dependencies import version
from .scan_heads import predicate


def waiting(db, selected, config):
    settings = config.data['sapiensid']
    options = settings['consolidation']
    if (not selected or options['scope']!='retained' or not settings['global_matching']['enabled']
            or db.execute('PRAGMA user_version').fetchone()[0]<17):
        return False
    from .body_identity import policy
    stamp = digest([policy(config),settings['global_matching']])
    return db.execute("""SELECT 1 FROM pending_body_merges p JOIN scans s ON s.id=?
        AND s.scope=p.scope AND s.namespace=p.namespace WHERE p.policy=? AND json_array_length(p.scenes)>?
        LIMIT 1""",(selected[0]['scan'],stamp,options['max_scenes'])).fetchone() is not None


def ready(db,config,scope,namespace,stamp,people,scenes,selection):
    from .body_identity import policy
    from .body_observations import contract
    from .scanning import scan_identity
    from .dependencies import store
    from .solver_execution import current
    options = config.data['sapiensid']['consolidation']
    allowed = set(options['scene_ids'])
    limit = options['max_scenes']
    if (current() is not None or options['scope']!='retained' or len(scenes)<=limit or
            (allowed and not (scenes-selection)<=allowed)):
        return False
    owners = sorted(people)
    scope_key = digest([scope,namespace,stamp,owners,sorted(scenes)])
    owner_state = [(p,version(db,'owner',p)) for p in owners]
    expected = digest([scope_key,owner_state])
    data_policy = scan_identity(config,namespace)
    body_contract, member_policy = contract(config), policy(config)
    checked = 0
    pending = False

    def heads():
        return db.execute("SELECT s.scene,s.id,s.total,COALESCE(v.version,0) FROM scans s "
            "LEFT JOIN evidence_versions v ON v.kind='scan' AND v.key=s.id "
            "WHERE s.scope=? AND s.namespace=? AND s.scene IN (SELECT value FROM json_each(?)) "
            "AND s.status='complete' AND s.cursor=s.total AND s.policy_sha=? AND "+predicate()+
            " ORDER BY s.scene",(scope,namespace,encoded(sorted(scenes)),data_policy))

    def valid(key,scan,revision):
        return db.execute('SELECT 1 FROM evidence_fingerprints WHERE kind=? AND key=? AND policy=? AND version=? AND digest=?',
            ('retained-alias-stage',key,expected,revision,scan_signature(scan))).fetchone() is not None

    def scan_signature(scan):
        return digest([scan,version(db,'raw-observations',scan),version(db,'raw-body',scan)])

    found = set()
    for scene,scan,total,revision in heads():
        found.add(scene)
        key = digest([scope_key,scene])
        if valid(key,scan,revision):
            continue
        if checked>=limit:
            pending = True
            continue
        checked += 1
        complete = db.execute('SELECT COUNT(*) FROM body_cues WHERE scan=? AND contract=?',(scan,body_contract)).fetchone()[0]
        stale = db.execute("SELECT 1 FROM body_observations b LEFT JOIN body_members m ON m.observation=b.id "
            "WHERE b.scan=? AND b.contract=? AND b.rejected IS NULL AND b.embedding IS NOT NULL AND m.policy IS NOT ? LIMIT 1",
            (scan,body_contract,member_policy)).fetchone()
        if complete!=total or stale:
            return False
        store(db,'retained-alias-stage',key,expected,revision,scan_signature(scan))
    if pending or found!=scenes or not db.in_transaction:
        return False
    # The caller holds the write transaction until aliases and publication debt
    # commit together. Compare every dependency again before authorizing the write.
    return (all(version(db,'owner',p)==clock for p,clock in owner_state) and
            sum(valid(digest([scope_key,scene]),scan,revision) for scene,scan,_,revision in heads())==len(scenes))
