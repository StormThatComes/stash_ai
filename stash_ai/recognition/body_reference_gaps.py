"""Retained identities with obsolete inputs are unknown rivals, never absent ones."""
from .body_observations import compatible_contracts
from .database import encoded
from .scan_heads import predicate as current_scan


def protect(db,config,scope,namespace,result):
    pending=[r[0] for r in db.execute('SELECT DISTINCT s.scene FROM body_observations b '
        'JOIN body_members m ON m.observation=b.id JOIN scans s ON s.id=b.scan '
        "WHERE s.scope=? AND s.namespace=? AND s.status='complete' AND m.person IS NOT NULL "
        "AND m.status IN ('accepted','provisional') AND m.role='anchor' AND b.rejected IS NULL "
        'AND b.contract NOT IN (SELECT value FROM json_each(?)) AND '+current_scan()+
        ' ORDER BY CAST(s.scene AS INTEGER),s.scene',(scope,namespace,encoded(compatible_contracts(config))))]
    if pending:
        result['incompatible_body_reference_scenes']=pending
        for decision in result['decisions']:
            if decision['outcome']=='merge':
                decision.update(outcome='blocked',blockers=['incompatible_body_reference_contract'],required_scenes=pending)
    return result
