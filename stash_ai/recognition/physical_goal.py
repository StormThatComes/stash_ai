"""Opt-in face/body addresses for one frozen physical vote, never extra support."""
from collections import defaultdict
from .database import encoded


def freeze(db, config, goals):
    from .body_identity import enabled, available
    from .body_observations import contract
    if not enabled(config) or not available(db):
        goals['physical_alternatives'] = {}
        return
    ids = sorted({r['id'] for g in goals['groups'] for m in g['members'] for r in m['observations']})
    alternatives = defaultdict(list)
    for row in db.execute('SELECT b.id,b.face FROM body_observations b '
        'JOIN observations o ON o.id=b.face AND o.scan=b.scan '
        'JOIN body_members m ON m.observation=b.id JOIN members f ON f.observation=b.face '
        "WHERE m.status='accepted' AND m.person=f.person AND b.rejected IS NULL "
        'AND b.embedding IS NOT NULL AND b.contract=? '
        'AND b.face IN (SELECT value FROM json_each(?)) ORDER BY b.id', (contract(config), encoded(ids))):
        alternatives[row['face']].append(row['id'])
    goals['physical_alternatives'] = dict(alternatives)
    for group in goals['groups']:
        evidence = group['identity_evidence']
        collections = [evidence['core_units'], evidence['soft_units']]
        collections += [u for scene in evidence['scenes'].values() for u in scene.values()]
        collections += [m['identity_evidence'][k] for m in group['members'] for k in ('core_units','soft_units')]
        seen = set()
        for units in collections:
            for unit in units:
                if id(unit) in seen:
                    continue
                seen.add(id(unit))
                existing = {r['id'] for r in unit['observations']}
                additions = {oid: r['scene'] for r in unit['observations']
                             for oid in alternatives[r['id']] if oid not in existing}
                unit['observations'].extend(dict(id=oid, scene=scene) for oid, scene in sorted(additions.items()))
