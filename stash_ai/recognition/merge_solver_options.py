"""Bound the historical option pool without discarding live parent settings."""
from .merge_report_policy import parameter


def compact(config, previous, fresh, *, parents=(), limit=128):
    from .merge_solver_search import spread
    result = {}
    for key in sorted(previous.keys() | fresh.keys()):
        current = parameter(config.data, key)
        old, new = set(previous.get(key, ())), set(fresh.get(key, ()))
        values = old | new
        center = current if current is not None else min(values, default=0)
        if len(values) <= limit:
            result[key] = sorted(values, reverse=True)
            continue
        protected = {p[key] for p in parents if key in p}
        if current is not None:
            protected.add(current)
        protected.update((min(values), max(values)))
        # Keep both sides of each live setting before sampling stale boundaries.
        for value in tuple(protected):
            lower = [v for v in values if v < value]
            upper = [v for v in values if v > value]
            if lower:protected.add(max(lower))
            if upper:protected.add(min(upper))
        chosen = list(sorted(protected))
        # Fresh measurements take precedence over old trial-specific values;
        # spread preserves distant boundaries instead of only dense local values.
        from itertools import chain
        for value in chain(spread(new, center), spread(old-new, center)):
            if len(chosen) >= max(limit, len(protected)):
                break
            if value not in protected:
                chosen.append(value)
                protected.add(value)
        result[key] = sorted(chosen, reverse=True)
    return result
