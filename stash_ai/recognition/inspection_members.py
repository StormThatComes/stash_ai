"""Metadata-only member roles matching the identity report's assignment rules."""
from .decision_evidence import REASONS
from .inspection_policy import selection
from .reference_roles import observation_sql


def query(person, policy, kinds, *, filters, accepted_only, config):
    ctes, queries, args = [], [], []
    for kind in kinds:
        body = kind == 'body'
        observations, members = ('body_observations', 'body_members') if body else ('observations', 'members')
        status = 'm.status' if body else "'accepted'"
        base = (f"SELECT '{kind}' AS kind,o.id,s.scene,o.cue,o.seconds,o.scan,{status} AS status"
                + (',m.role' if body else '') + f' FROM {members} m '
                f'JOIN {observations} o ON o.id=m.observation JOIN scans s ON s.id=o.scan '
                'WHERE m.person=? AND s.scope=? AND s.namespace=?')
        current, current_args = selection(config, kind)
        base += ' AND ' + current
        base_args = [person['id'], person['scope'], person['namespace'], *current_args]
        if accepted_only and body:
            base += " AND m.status='accepted'"
        for column, value in zip(('s.scene', 'o.cue', 'o.id'), filters):
            if value is not None:
                base += f' AND {column}=?'
                base_args.append(value)
        if body:
            queries.append("SELECT kind,id,scene,cue,seconds,status,role,CASE WHEN status='accepted' THEN role ELSE status END AS state FROM ("+base+')')
            # All CTE parameters precede the final UNION parameters.
            body_args = base_args
            continue
        ctes.extend((
            'family(id) AS (SELECT ? UNION SELECT m.source FROM identity_merges m JOIN family f ON m.target=f.id '
            'JOIN people p ON p.id=m.source WHERE p.scope=? AND p.namespace=?)',
            'face_members AS ('+base+')'))
        args.extend((person['id'], person['scope'], person['namespace'], *base_args))
        categories = "CASE json_extract(c.payload,'$.reason') " + ' '.join(
            f"WHEN '{reason}' THEN '{category}'" for reason, (category, _) in REASONS.items()) + " ELSE 'unknown' END"
        ctes.append('supporting AS (SELECT o.id,c.policy,'+categories+' AS category,'
            'MAX(c.policy=?) OVER (PARTITION BY o.id) AS has_current FROM face_members o '
            f'JOIN candidates c ON {observation_sql("c")}=o.id AND c.scan=o.scan '
            "WHERE c.status='accepted' AND c.person IN (SELECT id FROM family))")
        ctes.append('categories AS (SELECT id,CASE WHEN COUNT(DISTINCT category)=1 THEN MIN(category) '
            "ELSE 'unknown' END AS category FROM supporting WHERE has_current=0 OR policy=? GROUP BY id)")
        args.extend((policy, policy))
        # Deliberately keep a missing or mixed category unknown, like _assignment.
        role = ("CASE WHEN category LIKE 'anchor%' THEN 'anchor' WHEN category LIKE 'auxiliary%' THEN 'auxiliary' "
                "WHEN category IN ('hybrid-attachment','body-merge-attachment') THEN 'attachment' ELSE 'unknown' END")
        queries.append('SELECT o.kind,o.id,o.scene,o.cue,o.seconds,o.status,'+role+' AS role,'+role+
                       ' AS state FROM face_members o LEFT JOIN categories r ON r.id=o.id')
    if 'body' in kinds:
        args.extend(body_args)
    return (('WITH RECURSIVE '+','.join(ctes)+' ' if ctes else '') + 'SELECT * FROM ('+
            ' UNION ALL '.join(queries)+')', args)
