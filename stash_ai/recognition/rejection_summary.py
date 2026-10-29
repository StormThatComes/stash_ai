"""Separate face rejection, independent body rejection and unavailable projections."""
from .person_signal import contract
from .body_observations import contract as body_contract,enabled as bodies_enabled


def summary(db,scan,config):
    face=dict(db.execute('SELECT rejected,COUNT(*) FROM observations WHERE scan=? '
                         'AND rejected IS NOT NULL GROUP BY rejected',(scan,)))
    body={};unavailable={}
    if config.data['sapiensid']['enabled']:
        unavailable=dict(db.execute("SELECT COALESCE(json_extract(p.metadata,'$.identity_rejection'),'unavailable'),COUNT(*) "
            "FROM person_evidence p JOIN observations o ON o.id=p.observation WHERE o.scan=? "
            "AND p.contract=? AND p.status='unavailable' GROUP BY 1",(scan,contract(config))))
    if bodies_enabled(config):
        body=dict(db.execute('SELECT rejected,COUNT(*) FROM body_observations WHERE scan=? '
            'AND contract=? AND rejected IS NOT NULL GROUP BY rejected',(scan,body_contract(config))))
    return dict(face=face,body=body,person_evidence_unavailable=unavailable)
