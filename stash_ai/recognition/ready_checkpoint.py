"""Final derived plan; source catalogue, history and bindings guard reuse."""
from types import SimpleNamespace


def encode(database, mapping, history):
    return dict(mapping=mapping, actions=list(history.actions),
                corroboration_pending=database._rebuild_corroboration_pending,
                optional_budget_fallback=getattr(database,'_rebuild_optional_budget_fallback',False),
                corroborated_plan=sorted(database._rebuild_corroborated_plan))


def validate(db, payload):
    if set(payload) != {'mapping','actions','corroboration_pending','corroborated_plan','optional_budget_fallback'}:
        raise ValueError('Invalid final checkpoint fields.')
    mapping = payload['mapping']
    if not isinstance(mapping,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in mapping.items()):
        raise ValueError('Invalid final checkpoint bindings.')
    if (not isinstance(payload['actions'],list) or type(payload['corroboration_pending']) is not bool
            or type(payload['optional_budget_fallback']) is not bool):
        raise ValueError('Invalid final checkpoint actions.')
    if not isinstance(payload['corroborated_plan'],list) or any(not isinstance(v,str) for v in payload['corroborated_plan']):
        raise ValueError('Invalid final checkpoint corroboration.')
    for row in db.execute('SELECT DISTINCT person FROM members'):
        if row[0] not in mapping:
            raise ValueError('Incomplete final checkpoint bindings.')
    for action in payload['actions']:
        if (not isinstance(action,dict) or action.get('kind') not in ('strict_component','verified_attachment')
                or not isinstance(action.get('source'),str) or not isinstance(action.get('target'),str)
                or not isinstance(action.get('proof'),dict)):
            raise ValueError('Invalid final reference action.')


def restore(database, db, payload):
    validate(db,payload)
    database._rebuild_corroboration_pending = payload['corroboration_pending']
    database._rebuild_optional_budget_fallback = payload['optional_budget_fallback']
    database._rebuild_corroborated_plan = set(payload['corroborated_plan'])
    return payload['mapping'], SimpleNamespace(actions=tuple(payload['actions']))
