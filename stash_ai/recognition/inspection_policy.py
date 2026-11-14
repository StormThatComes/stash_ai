"""One definition of current evidence for inspector lists and direct navigation."""
from .body_observations import contract
from .models import namespace
from .scan_heads import predicate
from .scanning import scan_identity


def selection(config, kind, *, scan='s', observation='o'):
    """Never fall back to an older scan or a different collection configuration."""
    sql = predicate(scan) + f' AND {scan}.policy_sha=?'
    args = [scan_identity(config, namespace(config))]
    if kind == 'body':
        sql += f' AND {observation}.contract=?'
        args.append(contract(config))
    return sql, args
