"""Final frozen repairs can alias references, never train new appearances."""
from dataclasses import replace

from .clustering import _groups, matching_policy
from .config import plain
from .global_views import catalogue, consolidation_plan, merge_views
from .identity_merges import canonical
from .reference_usage import ReferenceUsage
from .strict_partition import summary_views


def finalize(database, config, mapping, history):
    if not config.data['clustering']['anchor_view_matching']:
        return
    db = database.connection
    if not db.execute("SELECT 1 FROM candidates WHERE json_extract(payload,"
            "'$.global_view_consolidation.blocker')='incomplete_strong_component' LIMIT 1").fetchone():
        return
    first = db.execute('SELECT scope,namespace FROM scans LIMIT 1').fetchone()
    if first is None:
        return
    usage = ReferenceUsage(config.data['clustering']['max_reference_bytes'])
    groups = _groups(db, *first, config.data['clustering'], usage=usage)
    local = {g.identifier for g in groups}
    # Unselected historical identities stay competitors. Repairing them requires
    # a replay including their source scenes, rather than a partial strict bank.
    protected = {g.identifier for g in groups if mapping.get(g.identifier) in history.people}
    from .reference_roles import strict_sql
    protected.update(r[0] for r in db.execute('SELECT DISTINCT m.person FROM members m '
        'JOIN observations o ON o.id=m.observation WHERE NOT (' + strict_sql() + ')'))
    external = tuple(g for g in history.groups if g.identifier not in local)
    summaries = summary_views(tuple(groups)+external, usage)
    views = catalogue(db, config, *first, usage=usage, complete=True)
    views += tuple(v for v in history.views if v.authority == 'strict' and v.group.identifier not in local)
    strict_data = plain(config.data)
    strict_data['recognition']['min_merge_support'] = 1
    strict_data['clustering'].update(view_match_threshold=strict_data['recognition']['match_threshold'],
        view_cohesion_threshold=strict_data['clustering']['cohesion_threshold'],
        view_ambiguity_margin=strict_data['recognition']['ambiguity_margin'])
    strict = replace(config, data=strict_data)
    proposals = []
    for bank, policy in ((summaries, strict), (views, config)):
        for decision in consolidation_plan(db, bank, policy):
            people = set(decision['people'])
            if (decision['outcome'] == 'merge' and decision['partition_strategy'] in ('reciprocal_supported_pair','supported_complete_subgroup')
                    and people <= local and not people.intersection(protected)):
                proposals.append((bank, policy, decision))
    # Different frozen routes must not claim overlapping incompatible repairs.
    conflicted = set()
    for _, _, left in proposals:
        for _, _, right in proposals:
            if left['people'] != right['people'] and set(left['people']).intersection(right['people']):
                conflicted.update(left['people']);conflicted.update(right['people'])
    applied = set()
    for bank, policy, decision in proposals:
        pair = decision['people']
        if pair in applied or conflicted.intersection(pair):
            continue
        merge_views(database, bank, policy, matching_policy(config), plan=(decision,),
                    evidence_field='final_subcomponent_merge')
        applied.add(pair)
        target = canonical(db, pair[0])
        # Binding already selected stable labels. Keep the oldest real label,
        # using the deterministic replay order only for newly born identities.
        def age(person):
            prior = db.execute('SELECT created FROM old_people WHERE id=?', (mapping[person],)).fetchone()
            return (0, prior[0], mapping[person]) if prior else (1, db.execute(
                'SELECT created FROM people WHERE id=?', (person,)).fetchone()[0], mapping[person])
        survivor = min(pair, key=age)
        mapping[target] = mapping[survivor]
    return bool(applied)
