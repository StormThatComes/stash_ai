"""Read-only, bounded cross-scene comparisons, including withheld anchors.

No labels are inferred from a high score alone. Report only numeric evidence
and explicit blockers; raw face embeddings never appear in JSON output.
"""
from .clustering import _similarities, vector
from .pose_policy import anchor_sql
from .replay_addresses import address


def _rows(db, scan, policy, clustering, limit=256):
    anchor = anchor_sql('o', clustering)
    rows = db.execute(
        f'SELECT o.id,o.scan,o.cue,o.seconds,o.geometry,o.dimensions,o.embedding,'
        'c.status,c.person,'
        "json_extract(c.payload,'$.reason') AS reason "
        f'FROM observations o JOIN candidates c ON '
        f'{address(db, "c")}=o.id AND c.policy=? '
        f'WHERE o.scan=? AND o.rejected IS NULL AND {anchor} '
        'ORDER BY o.cue,o.seconds,o.id LIMIT ?',
        (policy, scan['id'], limit+1)).fetchall()
    return rows[:limit], len(rows)>limit


def compare(db, origin, others, policy, config, *, limit=20):
    """Compare selected scene to other scenes under the same committed policy.

    Output contains both all-vs-all anchor nearest neighbours (including
    withheld) and aggregated matches against the *confirmed* target identities.
    """
    from .fragmentation_report import _representatives, _scene_identities, _selected_scan
    from .models import namespace
    from .clustering import Group
    settings, grouping = config.data['recognition'], config.data['clustering']
    source_rows, src_truncated = _rows(db, origin, policy, grouping)
    source_vectors = [(row, vector(row)) for row in source_rows]
    results = []
    for scene_id in others:
        other = _selected_scan(db, str(scene_id), namespace(config), policy,
                               scope=origin['scope'])
        if other['id']==origin['id']:
            raise ValueError('Compare-scene-id must refer to a different scene.')
        target_rows, tgt_truncated = _rows(db, other, policy, grouping)
        target_vectors = [(row, vector(row)) for row in target_rows]
        people = _scene_identities(db, other['id'], policy)
        known = [(p, Group(p['identity_id'], tuple(v for _,v in _representatives(
            db,other,p['identity_id'],grouping['max_samples_per_person'],grouping,config=config)), True))
            for p in people]
        from .historical_references import History,rank
        from .global_views import catalogue
        from .reference_usage import ReferenceUsage
        from .scanning import scan_identity
        frozen=tuple(group for _,group in known if group.samples)
        usage=ReferenceUsage(grouping['max_reference_bytes'])
        for group in frozen:
            usage.add(group.samples,people=1,kind='Report strict')
        views=(catalogue(db,config,origin['scope'],origin['namespace'],usage=usage,
            scan_policy=scan_identity(config,origin['namespace']),complete=True) if grouping['anchor_view_matching'] else ())
        if grouping['view_prototypes_enabled'] and grouping['auxiliary_matching']:
            from .verified_views import historical_views
            views=tuple(views)+historical_views(db,config,origin['scope'],origin['namespace'],[],frozen,usage)
        owners={group.identifier for group in frozen}
        views=tuple(view for view in views if view.group.identifier in owners)
        history=History(frozen,views,{},'read-only-report')
        matches = []
        evidence = []
        threshold = settings['match_threshold']
        cohesion = grouping['cohesion_threshold']
        margin = settings['ambiguity_margin']
        for src, query in source_vectors:
            ranked = []
            for person, group in known:
                if not group.samples:
                    continue
                mean, low = _similarities(group, query)
                ranked.append((mean, low, person['identity_id'], person['performer_id']))
            ranked=[item for item in ranked if item[1]>=cohesion]
            ranked.sort(key=lambda row:(-row[0], row[2]))
            winner = ranked[0] if ranked else None
            runner = ranked[1] if len(ranked)>1 else None
            reason = ('no_confirmed_target_identity' if winner is None
                      else 'below_match_threshold' if winner[0]<threshold
                      else 'below_cohesion_threshold' if winner[1]<cohesion
                      else 'ambiguous_identity' if runner and winner[0]-runner[0]<margin
                      else 'numeric_match_only')
            decision=[]
            matched=rank([src],history,config,decisions=decision)
            production=(dict(matched[1],identity=matched[0]) if matched else
                        dict(outcome=decision[0]['outcome']) if decision else dict(outcome='no_eligible_match'))
            matches.append(dict(production_match=production,observation_id=src['id'],cue=src['cue'],
                status=src['status'],reason=src['reason'],best_person=winner[2] if winner else None,
                best_performer=winner[3] if winner else None,
                mean=round(winner[0],6) if winner else None,
                minimum=round(winner[1],6) if winner else None,
                next_mean=round(runner[0],6) if runner else None,
                verdict=reason))
            for tgt, target_vector in target_vectors:
                import numpy as np
                cosine = float(np.dot(query, target_vector))
                evidence.append((cosine,src,tgt))
        evidence.sort(key=lambda item:(-item[0],item[1]['id'],item[2]['id']))
        hits = [dict(cosine=round(cosine,6),
                    source=dict(observation_id=a['id'], cue=a['cue'], status=a['status'],
                                person=a['person'], reason=a['reason']),
                    target=dict(observation_id=b['id'], cue=b['cue'], status=b['status'],
                                person=b['person'], reason=b['reason']))
                for cosine,a,b in evidence[:limit]]
        outcomes={}
        for item in matches:
            outcomes[item['verdict']]=outcomes.get(item['verdict'],0)+1
        results.append(dict(scene_id=str(scene_id), scan_id=other['id'],
            reference_identities=len(known),
            anchors_scored={'source':len(source_vectors),'target':len(target_vectors),
                            'source_truncated':src_truncated,'target_truncated':tgt_truncated},
            source_verdicts=outcomes, per_observation=matches,
            nearest_anchor_pairs=hits, total_anchor_pairs=len(evidence),
            pairs_above_global_match=sum(item[0]>=threshold for item in evidence),
            limitations=[
                'production_match uses the production matcher restricted to target-scene identities; global competition may differ.',
                'Numeric scores do not establish identity or authorize automatic merging.',
                'Only the first 256 anchor observations per scene are compared in canonical cue order.',
                'Nearest anchor pairs include withheld candidates but do not imply confirmed identities.',
                'Existing target identity references may also come from scenes outside this comparison.',
                'Co-presence, original media, provenance and replay ordering are not revalidated.']))
    return results
