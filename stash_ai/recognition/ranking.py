"""Stable, bounded reductions over every eligible identity rival."""


def best_two(ranked, candidate):
    key = -candidate[0], candidate[1]
    if not ranked or key < (-ranked[0][0], ranked[0][1]):
        ranked.insert(0, candidate)
    elif len(ranked) == 1 or key < (-ranked[1][0], ranked[1][1]):
        ranked.insert(1, candidate)
    if len(ranked) > 2:
        ranked.pop()
