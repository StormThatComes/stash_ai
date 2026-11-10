"""One-to-one continuations for physically independent components of a cue pair."""
from collections import defaultdict

from .crowd_context import pair
from itertools import product


def components(edges):
    neighbours = defaultdict(set)
    for left, right in edges:
        neighbours[left].add(right)
        neighbours[right].add(left)
    while neighbours:
        pending = [next(iter(neighbours))]
        members = set()
        while pending:
            member = pending.pop()
            if member not in members:
                members.add(member)
                pending.extend(neighbours.pop(member))
        yield members


def separated(ids, geometry):
    from itertools import combinations
    for a, b in combinations(ids, 2):
        first, second = geometry[a], geometry[b]
        x,y,r,t = first['bounds']
        u,v,s,w = second['bounds']
        if min(r,s)>max(x,u) and min(t,w)>max(y,v) and pair([first,second]) is None:
            return False
    return True


def joint_links(edges, physical, threshold, margin):
    by_left = defaultdict(dict)
    for (left, right), score in edges.items():
        by_left[left][right] = score
    for members in components(edges):
        left = sorted(members & by_left.keys())
        right = members.difference(left)
        if max(len(left), len(right)) > 4:
            continue
        if not separated(left, physical) or not separated(right, physical):
            continue
        domains = [[(None, threshold-margin), *sorted(by_left[a].items())] for a in left]
        # At most 5^4 assignments. For each best edge, certify the best
        # complete assignment without it, including the option to abstain.
        solutions = []
        for choices in product(*domains):
            targets = [target for target,_ in choices if target is not None]
            if len(targets)==len(set(targets)):
                solutions.append((sum(score for _,score in choices),choices))
        best, choices = max(solutions,key=lambda entry:entry[0])
        for index,(target,score) in enumerate(choices):
            if target is None or score<threshold:
                continue
            alternative = max(total for total,others in solutions if others[index][0]!=target)
            if best-alternative+1e-12>=margin and best>alternative:
                yield left[index],target,score
