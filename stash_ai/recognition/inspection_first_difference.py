"""Earliest recorded difference in scene/cue order and its recorded dependants."""
from collections import defaultdict, deque


def order(row):
    return int(row['scene']), row['cue']


def changed(key, before, after):
    if before['owners'][key] != after['owners'][key]:
        return True
    a, b = before['decisions'].get(key, {}), after['decisions'].get(key, {})
    return any(a.get(field) != b.get(field) for field in
               ('role','route','stage','signal','links','references','reference_sources'))


def first_difference(rows, before, after):
    differences = {key for key in rows if changed(key, before, after)}
    if not differences:
        return dict(first=[],later=[],changed=differences)
    earliest = min(order(rows[key]) for key in differences)
    first = sorted(key for key in differences if order(rows[key]) == earliest)
    # Follow each actual reference graph independently; edges from different
    # trials must never be combined into an apparent chain of evidence.
    linked = set()
    for trial in (before, after):
        children = defaultdict(list)
        for key, d in trial['decisions'].items():
            for ref in d.get('references', ()):
                children[ref].append(key)
        reached = set(first)
        queue = deque(first)
        while queue:
            for child in children.get(queue.popleft(), ()):
                if child not in reached:
                    reached.add(child)
                    queue.append(child)
        linked.update(reached)
    later = sorted((k for k in differences.intersection(linked) if order(rows[k]) > earliest),
                 key=lambda k: (order(rows[k]), k))
    return dict(first=first,later=later,changed=differences)


def stage_finding(result):
    info = result['intervention']
    enabled, omitted = (result['trials'][k] for k in ('enabled','omitted'))
    if not info['omitted_visits']:
        return 'The selected stage was not reached in this trial.'
    if enabled['success'] and not omitted['success']:
        return 'The selected stage was necessary for the complete goal on this selection.'
    if enabled['success'] and omitted['success']:
        return 'The complete goal also succeeded with the selected stage omitted.'
    if omitted['success']:
        return 'Only the trial omitting this stage met the complete goal.'
    return 'Neither trial met the complete goal. Inspect the first differing cue and residual observations.'
