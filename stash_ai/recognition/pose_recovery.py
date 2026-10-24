"""Final frozen query recovery; neither route may train another appearance."""
from .clustering import matching_policy, vector, co_presence_allows
from .database import encoded
from .historical_references import History, capture, rank, rank_many
from .reference_roles import observation_sql
from .reference_usage import ReferenceUsage
from .pose_policy import query_allows, is_anchor
from .view_prototypes import pose_matches, pose_view
from .frozen_ownership import FrozenOwnership
from .progress import tracking


def recover(durable, database, config, external):
    return _recover(durable, database, config, external, appearance=False)


def recover_appearances(durable, database, config, external):
    if config.data['clustering']['strict_appearance_matching'] or config.data['clustering'].get('anchor_outlier_quarantine'):
        return _recover(durable, database, config, external, appearance=True)


def recover_joint(durable, database, config, external):
    if config.data['clustering']['strict_appearance_matching'] or config.data['clustering'].get('anchor_outlier_quarantine'):
        return _recover(durable, database, config, external, appearance=True, relaxed=True)


def freeze(database, config, external):
    db=database.connection
    first=db.execute('SELECT scope,namespace FROM scans LIMIT 1').fetchone()
    selected=capture(db, [], config, *first, 'pre-final-merge', resolve_aliases=False)
    return History(selected.groups+external.groups,
        tuple(v for v in selected.views+external.views if v.authority in ('strict','quarantined_core')), {}, 'pre-final-merge')


def recover_final(durable, database, config, external, bank):
    # Preserve each original set of witnesses; only labels may change. Never
    # rebuild centroids/views from the newly merged or recovered membership.
    from dataclasses import replace
    from .identity_merges import canonical
    alias=lambda p:canonical(database.connection,p)
    bank=replace(bank,groups=tuple(replace(g,identifier=alias(g.identifier)) for g in bank.groups),
        views=tuple(replace(v,group=replace(v.group,identifier=alias(v.group.identifier))) for v in bank.views))
    return _recover(durable,database,config,external,appearance=config.data['clustering']['strict_appearance_matching'] or config.data['clustering'].get('anchor_outlier_quarantine'),
                    relaxed=True,final_bank=bank)


def _recover(durable, database, config, external, *, appearance, relaxed=False, final_bank=None):
    grouping = config.data['clustering']
    field = 'strict_appearance' if appearance else 'pose_recovery'
    prefix = 'strict_appearance' if appearance else 'pose_uncertain'
    if final_bank is not None:field=prefix='final_recovery'
    if not grouping['anchor_view_matching']:
        return
    db = database.connection
    first = db.execute('SELECT scope,namespace FROM scans LIMIT 1').fetchone()
    if first is None:
        return
    if not db.execute("SELECT 1 FROM candidates WHERE status='withheld' LIMIT 1").fetchone():
        return
    usage = ReferenceUsage(grouping['max_reference_bytes'])
    if final_bank is None:
        selected = capture(db, [], config, *first, field+'-roots', resolve_aliases=False)
        from .verified_views import historical_views
        verified = (historical_views(db, config, *first, [], selected.groups, usage, selected.confirmations)
                    if grouping['view_prototypes_enabled'] and grouping['auxiliary_matching'] else ())
        views = tuple(v for v in selected.views+verified+external.views if v.authority in ('strict', 'verified_view', 'quarantined_core'))
        bank = History(selected.groups+external.groups, views, {}, 'frozen-'+field)
    else:
        bank=final_bank;views=bank.views
    if not views and final_bank is None:
        return
    for group in bank.groups:
        usage.add(group.samples, kind='Pose recovery strict rivals')
    for view in views:
        usage.add(view.examples, kind='Pose recovery view witnesses')
    from .identity_merges import canonical
    aliases=dict(external.aliases)
    aliases.update({r[0]:canonical(db,r[0]) for r in db.execute('SELECT id FROM people')})
    ownership = FrozenOwnership(db, durable.connection, aliases)
    policy = matching_policy(config)
    db.execute('CREATE TEMP TABLE pose_recovery_claims(candidate TEXT PRIMARY KEY,observation TEXT,owner TEXT,proof TEXT)')
    try:
        rows = db.execute('SELECT o.*,c.id AS candidate FROM candidates c JOIN observations o ON o.id=' +
            observation_sql('c') + " WHERE c.policy=? AND c.status='withheld' AND o.rejected IS NULL ORDER BY o.id", (policy,))
        from .late_recovery import ranked_queries
        ranked = ranked_queries(rows, bank, config, appearance=appearance, relaxed=relaxed)
        with tracking(ranked, 'Match supported strict appearances' if appearance else 'Check uncertain face pose', unit='observation') as tracked:
            for row, ordinary, fallback in tracked:
                if not grouping['auxiliary_matching'] and not is_anchor(row['quality'], grouping):
                    continue
                if not query_allows(row['quality'], grouping):
                    continue
                # Only queries without an ordinary positive match enter a
                # fallback; existing support/conflict decisions stay unchanged.
                if ordinary is not None and final_bank is None:
                    continue
                match = ordinary or fallback
                if match is None:
                    continue
                owner, proof = match
                chosen = [v for v in views if v.group.identifier == owner and any(
                    p['medoid']==v.medoid and p['authority']==v.authority for p in proof['view_evidence'])]
                crossed = [v for v in chosen if not pose_matches(row['geometry'], v.pose)]
                if (final_bank is None and not appearance and not crossed) or not ownership.proof_allows(owner, proof, bank.groups):
                    continue
                references = bank.groups+tuple(v.group for v in views)
                candidates=([v.group for v in chosen] if proof['mode']=='view' else
                            [g for g in bank.groups if g.identifier==owner])
                if not any(co_presence_allows(db, row, g, references, vector(row), config) for g in candidates):
                    continue
                proof = dict(proof, joint_pose_appearance=appearance and relaxed, post_merge=final_bank is not None,
                    query_pose=pose_view(row['geometry']),
                    crossed_poses=sorted({v.pose for v in crossed}), minimum_query_support=max(2, grouping['view_attach_min_support']))
                db.execute('INSERT INTO pose_recovery_claims VALUES (?,?,?,?)', (row['candidate'], row['id'], owner, encoded(proof)))
        for row in db.execute('SELECT * FROM pose_recovery_claims'):
            if row['owner'] in external.people:
                db.execute('INSERT OR IGNORE INTO people VALUES (?,?,?,?,?,?)', external.people[row['owner']])
            db.execute("UPDATE candidates SET person=?,status='pending',payload=json_set(payload,"
                f"'$.prior_reason',json_extract(payload,'$.reason'),'$.reason','{prefix}_pending',"
                f"'$.{field}',json(?)) WHERE id=?", (row['owner'], row['proof'], row['candidate']))
        from .decision_resolution import resolve_pending
        resolve_pending(db, policy, config)
        from .query_support import eligible
        stream = lambda: db.execute('SELECT o.id,o.scan,o.seconds,o.dimensions,o.embedding,c.person,'
            f"json_extract(c.payload,'$.{field}.medoids') AS medoids FROM candidates c "
            'JOIN observations o ON o.id=' + observation_sql('c') +
            f" WHERE c.policy=? AND c.status='pending' AND json_extract(c.payload,'$.reason')='{prefix}_pending' "
            'ORDER BY o.embedding,o.geometry,o.id', (policy,))
        supported = eligible(stream, config, key=lambda r:r['person'], prototype=lambda r:r['medoids'],
                             minimum=lambda _:max(2, grouping['view_attach_min_support']))
        db.execute(f"UPDATE candidates SET status='accepted',payload=json_set(payload,'$.reason','{prefix}_identity') "
            f"WHERE policy=? AND status='pending' AND json_extract(payload,'$.reason')='{prefix}_pending' "
            'AND ' + observation_sql('candidates') + ' IN (SELECT value FROM json_each(?))', (policy, encoded(sorted(supported))))
        db.execute('INSERT OR IGNORE INTO members SELECT '+observation_sql('c')+',c.person FROM candidates c '
            f"WHERE policy=? AND status='accepted' AND json_extract(payload,'$.reason')='{prefix}_identity'", (policy,))
        db.execute("UPDATE candidates SET person=NULL,status='withheld',payload=json_set(payload,'$.reason',"
            f"'{prefix}_insufficient_support') WHERE policy=? AND status='pending' "
            f"AND json_extract(payload,'$.reason')='{prefix}_pending'", (policy,))
    finally:
        db.execute('DROP TABLE pose_recovery_claims')
