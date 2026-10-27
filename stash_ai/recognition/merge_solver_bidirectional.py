"""Early permissive experiments and an independent frontier for repairing them.

This frontier may retain cross-set merges as search parents. It never selects
the exported winner or certifies an interpolated/frozen-evidence proposal.
"""
from collections import deque

from .config import digest
from .merge_report_policy import parameter, patched, proposal
from .merge_solver_constraints import joint_separation
from .merge_solver_search import PREFIX, interleave


def relaxed_seeds(config, feedback):
    """Joint jumps across matching gates; acquisition/model contracts stay fixed."""
    hybrid = {PREFIX + key: min(parameter(config.data, PREFIX + key), value)
              for key, value in dict(face_floor=.2, person_threshold=.35,
                                     ambiguity_margin=0., min_support=1).items()}
    face = {key: min(parameter(config.data, key), value) for key, value in {
        'recognition.match_threshold': .35, 'clustering.cohesion_threshold': .25,
        'clustering.view_match_threshold': .35, 'clustering.view_cohesion_threshold': .25,
        'recognition.cooccurrence_merge_threshold': .35,
        'recognition.ambiguity_margin': 0., 'clustering.view_ambiguity_margin': 0.,
        'recognition.min_identity_support': 1, 'recognition.min_merge_support': 1,
        'clustering.view_min_support': 2, 'clustering.view_attach_min_support': 1,
        'clustering.same_scene_min_support': 1, 'clustering.auxiliary_min_support': 1,
    }.items()}
    # The first quality proposal is the joint relaxation of measured retained
    # admission gates. Missing measurements never become fabricated evidence.
    quality = next(iter(feedback.get('proposals', {}).get('face', ())), {})
    if config.data['sapiensid']['enabled']:
        yield dict(hybrid, **quality)
    yield dict(face, **quality)
    if config.data['sapiensid']['enabled']:
        yield dict(face, **hybrid, **quality)


def progress_rank(status):
    """Continuous within-set progress, including when complete stays at 1/N."""
    groups = status['groups']
    pairs = [g['pair_consolidation_fraction'] for g in groups if g['total_performers'] > 1]
    shared = [g['consolidation_fraction'] for g in groups]
    assigned = [g['assigned_fraction'] for g in groups]
    return (min(pairs, default=0.), sum(pairs)/max(1, len(pairs)),
            min(shared), sum(shared)/len(shared), min(assigned),
            sum(assigned)/len(assigned), -status['cross_set_observation_pairs'])


def expansion_candidates(config, changes, feedback):
    """A disjoint partial result needs outward moves, preserving its useful patch.

    Relax matching gates together and independently, beyond the fixed seed
    values. Current measurements also supply larger jumps. Production replay
    decides whether any of these moves helps; no monotonicity is assumed.
    """
    profiles = list(relaxed_seeds(config, {}))
    keys = set().union(*(p.keys() for p in profiles))
    active = feedback.get('provenance', {}).get('active_parameters', {})
    keys = sorted(keys, key=lambda k: (-active.get(k, 0), k))
    groups = [[k for k in keys if k.startswith(PREFIX) == hybrid] for hybrid in (True, False)]
    groups = sorted((g for g in groups if g), key=lambda g: -sum(active.get(k, 0) for k in g))

    def directed():
        for hints in feedback.get('proposals', {}).values():
            for hint in hints:
                patch = dict(changes)
                for key, value in hint.items():
                    current = changes.get(key, parameter(config.data, key))
                    if key in keys:
                        patch[key] = min(current, value)
                    elif key.startswith('clustering.') and type(current) in (int, float):
                        # Only measured retained admission gates, preserving
                        # any greater relaxation already found by the parent.
                        if key.endswith('pose_deviation'):
                            patch[key] = max(current, value)
                        elif key.startswith('clustering.min_'):
                            patch[key] = min(current, value)
                yield patch

    def joint():
        for fraction in (.9, .5, 0.):
            patch = {}
            for key in keys:
                current = changes.get(key, parameter(config.data, key))
                value = round(current*fraction, 6)
                if key.endswith('support'):
                    value = max(2 if key == 'clustering.view_min_support' else 1, int(value))
                patch[key] = value
            for group in groups:
                yield dict(changes, **{k: patch[k] for k in group})
            if len(groups) > 1:
                yield dict(changes, **patch)

    def coordinates():
        options = feedback.get('options', {})
        for key in keys:
            current = changes.get(key, parameter(config.data, key))
            lower = sorted({v for v in options.get(key, ()) if type(v) in (int, float) and v < current}, reverse=True)
            for value in lower[:1] + lower[-1:]:
                yield dict(changes, **{key: value})

    for patch in interleave((joint(), directed(), coordinates())):
        match = patch.get('recognition.match_threshold', parameter(config.data, 'recognition.match_threshold'))
        cohesion = patch.get('clustering.cohesion_threshold', parameter(config.data, 'clustering.cohesion_threshold'))
        if cohesion > match:
            patch['clustering.cohesion_threshold'] = match
        yield 'frontier_expansion', patch


def repair_candidates(config, changes, sources, safe):
    """Use witnesses and multiple routes back towards a completed disjoint state."""
    positive, negative = [], []
    for source in sources:
        query = source.get('query', {})
        positive.extend(query.get('positive_witnesses', ()))
        negative.extend(query.get('negative_witnesses', ()))
    joint = joint_separation(patched(config, changes), positive, negative)
    hints = list(joint['proposals'])
    for source in sources:
        hints.extend(p for values in source.get('proposals', {}).values() for p in values)

    def directed():
        for patch in hints:
            yield 'separation_repair', dict(changes, **patch)

    def bridges():
        keys = sorted(changes.keys() | safe.keys())
        # Joint and per-family bridges: tightening every gate together can lose
        # the good merge before the gate rejecting the unwanted merge is found.
        for fraction in (.5, .25, .75, 1.):
            joint = {}
            for key in keys:
                start = changes.get(key, parameter(config.data, key))
                end = safe.get(key, parameter(config.data, key))
                if start == end:
                    continue
                if type(start) is bool or start is None or end is None:
                    value = end
                elif key.endswith('support'):
                    value = int(round(start + (end-start)*fraction))
                else:
                    value = round(start + (end-start)*fraction, 6)
                joint[key] = value
            yield 'boundary_bridge', dict(changes, **joint)
            for hybrid in (True, False):
                yield 'boundary_bridge', dict(changes, **{
                    k: v for k, v in joint.items() if k.startswith(PREFIX) == hybrid})
        # Restore a single gate while keeping the other permissive gates open.
        for key in keys:
            patch = dict(changes, **{key: safe.get(key, parameter(config.data, key))})
            yield 'separation_repair', patch

    for strategy, patch in interleave((directed(), bridges())):
        match = patch.get('recognition.match_threshold', parameter(config.data, 'recognition.match_threshold'))
        cohesion = patch.get('clustering.cohesion_threshold', parameter(config.data, 'clustering.cohesion_threshold'))
        if cohesion > match:
            patch['clustering.cohesion_threshold'] = match
        yield strategy, patch


class BidirectionalSearch:
    """Bounded lane, independent of the export rank and the normal five-state beam."""
    width = 3

    def __init__(self, config, goals, maximum, baseline_feedback):
        self.config = config
        self.enabled = len(goals['groups']) > 1 and any(len(g['members']) > 1 for g in goals['groups'])
        self.budget = min(64, max(0, maximum-2), max(4, maximum//3)) if self.enabled else 0
        self.executed = self.offered = self.turn = 0
        self.last_trial = None
        self.frontier, self.streams = {}, {}
        self.safe = None
        self.baseline_feedback = baseline_feedback
        self.seeds = deque(relaxed_seeds(config, baseline_feedback)) if self.enabled else deque()
        self.seen = {digest({})}

    def observe(self, name, changes, status, rank, outcome, *, phase=None):
        if not self.enabled:
            return
        self.executed += int(phase == 'bidirectional')
        if status['disjoint'] and (self.safe is None or rank > self.safe['rank']):
            self.safe = dict(name=name, changes=changes, rank=rank)
        if name == 'baseline':
            return
        candidate = dict(changes=changes, rank=progress_rank(status) + rank,
                         outcome=outcome, disjoint=status['disjoint'])
        pool = dict(self.frontier, **{name: candidate})
        chosen, signatures = {}, set()
        for key in sorted(pool, key=lambda n: pool[n]['rank'], reverse=True):
            # Keep distinct parameter routes even after the same overmerge.
            signature = (pool[key]['outcome'], digest(pool[key]['changes']))
            if signature not in signatures:
                chosen[key] = pool[key]
                signatures.add(signature)
            if len(chosen) == self.width:
                break
        self.frontier = chosen
        safe_name = self.safe['name'] if self.safe else 'baseline'
        self.streams = {k: v for k, v in self.streams.items() if k[0] in chosen and
                        k[1] == (None if chosen[k[0]]['disjoint'] else safe_name)}

    def next(self, trial, feedback, *, solved=False, force=False):
        if (not self.enabled or solved or self.executed >= self.budget or
                (not force and (trial < 2 or trial % 2 or self.last_trial == trial))):
            return None
        self.last_trial = trial

        def seed():
            while self.seeds:
                yield 'baseline', 'relaxed_seed', self.seeds.popleft()

        def repairs():
            parents = list(self.frontier)
            if not parents:
                return
            winner, others = parents[:1], parents[1:]
            if others:
                offset = (self.turn//2) % len(others)
                others = others[offset:] + others[:offset]
            parents = winner + others if self.turn % 2 == 0 else others + winner
            self.turn += 1
            for parent in parents:
                state = self.frontier[parent]
                safe_name = self.safe['name'] if self.safe else 'baseline'
                key = parent, None if state['disjoint'] else safe_name
                if key not in self.streams:
                    candidates = (expansion_candidates(self.config, state['changes'], feedback[parent])
                                  if state['disjoint'] else repair_candidates(self.config,
                                      state['changes'], [feedback[parent], self.baseline_feedback],
                                      self.safe['changes'] if self.safe else {}))
                    self.streams[key] = iter(candidates)
                for strategy, patch in self.streams[key]:
                    yield parent, strategy, patch

        # Alternate broad starts and their repairs; normal trials occupy the
        # intervening slots. Duplicates do not consume a production replay.
        streams = (seed(), repairs()) if self.offered % 2 == 0 else (repairs(), seed())
        for stream in streams:
            for parent, strategy, changes in stream:
                patch = proposal(self.config, changes)
                key = digest(patch['yaml_patch'])
                if key in self.seen:
                    continue
                self.seen.add(key)
                self.offered += 1
                family = 'hybrid' if any(k.startswith(PREFIX) for k in changes) else 'face'
                return family, 'bidirectional', parent, strategy, changes
        return None
