"""Explicit, bounded full-replay neighbourhood validation."""
from .merge_report_policy import parameter
from .merge_solver_metrics import scale


def probes(config, changes, active, limit=8, *, options=None, groups=()):
    """Cover coordinates before their second side; interleave correlated probes."""
    from .config import digest
    from .merge_solver_domains import registry, normalize
    domains = registry(config)
    keys = sorted(changes, key=lambda k: (-active.get(k, 0), k))
    keys += sorted(set(active)-set(keys), key=lambda k: (-active[k], k))
    keys = [k for k in keys if k in domains]
    values = {}
    for key in keys:
        current = changes.get(key, parameter(config.data, key))
        if type(current) is bool:
            values[key] = [not current]
        elif type(current) in (int, float):
            step = 1 if domains[key].kind is int else .002*scale(key, current)
            nearby = []
            for sign in (-1, 1):
                measured = [v for v in (options or {}).get(key, ())
                            if type(v) in (int, float) and 0 < sign*(v-current) <= step]
                value = min(measured, key=lambda v: abs(v-current)) if measured else current+sign*step
                value = domains[key].project(value)
                if value != current and value not in nearby:
                    nearby.append(value)
            values[key] = nearby
    def singles():
        for side in range(2):
            for key in keys:
                if side < len(values.get(key, ())):
                    yield dict(changes, **{key: values[key][side]})
    def joints():
        coupled = tuple(groups)+(
            ('sapiensid.matching.face_floor', 'sapiensid.matching.person_threshold'),
            ('recognition.match_threshold', 'clustering.cohesion_threshold'),
            ('clustering.view_match_threshold', 'clustering.view_cohesion_threshold'))
        for group in coupled:
            selected = [k for k in group if values.get(k)][:3]
            if len(selected) < 2:
                continue
            for side in range(2):
                yield dict(changes, **{k: values[k][min(side, len(values[k])-1)] for k in selected})
    individual, joint = iter(singles()), iter(joints())
    seen = {digest(normalize(config, changes))}
    count = 0
    while individual is not None or joint is not None:
        for lane in ('single', 'single', 'joint'):
            stream = individual if lane == 'single' else joint
            if stream is None:
                continue
            try:
                patch = normalize(config, next(stream))
            except StopIteration:
                if lane == 'single':individual = None
                else:joint = None
                continue
            signature = digest(patch)
            if signature not in seen and count < limit:
                seen.add(signature)
                count += 1
                yield patch
            if count >= limit:
                return


def confidence(checks):
    # Positive preference requires at least two full replays with no loss.
    return int(len(checks)>=2 and all(checks))
