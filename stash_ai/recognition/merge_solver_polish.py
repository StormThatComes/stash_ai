"""Bounded full-replay ablation followed by local stability probes."""
from .config import digest
from .merge_report_policy import parameter
from .merge_solver_constraints import learned_groups
from .merge_solver_domains import normalize
from .merge_solver_robustness import probes
from .merge_solver_search import interleave


def allowance(maximum, changes, active):
    if maximum < 20:
        return 0
    # Reserve more work for a broad patch, never more than a quarter of replays.
    return min(24, maximum//4, max(4, len(changes)+min(8, len(active))))


def ablations(config, changes, active, groups=()):
    keys = sorted(changes, key=lambda k: (active.get(k, 0), k))
    removals = [(k,) for k in keys]
    removals += [tuple(k for k in keys if k.startswith(prefix))
                 for prefix in ('sapiensid.', 'recognition.', 'clustering.')]
    removals += list(groups)
    seen = set()
    for removed in removals:
        if not removed or not set(removed) <= changes.keys():
            continue
        patch = normalize(config, dict(changes, **{k: parameter(config.data, k) for k in removed}))
        signature = digest(patch)
        if len(patch) < len(changes) and signature not in seen:
            seen.add(signature)
            yield patch


class Polisher:
    def __init__(self, config, budget):
        self.config, self.budget = config, budget
        self.executed = self.minimized = 0
        self.parent = None
        self.ablations = iter(())
        self.validation = None

    def next(self, winner, parents, changes, feedback):
        if self.executed >= self.budget:
            raise StopIteration
        if self.validation is None:
            if self.parent != winner:
                self.parent = winner
                source = feedback.get(winner, {})
                self.ablations = iter(ablations(self.config, changes(winner),
                    source.get('provenance', {}).get('active_parameters', {}), learned_groups(source)))
            if self.minimized < self.budget//2:
                try:
                    return 'minimization', winner, 'parameter_ablation', next(self.ablations)
                except StopIteration:
                    pass
            def validate(parent):
                source = feedback.get(parent, {})
                for patch in probes(self.config, changes(parent),
                        source.get('provenance', {}).get('active_parameters', {}), limit=self.budget*2,
                        options=source.get('options'), groups=learned_groups(source)):
                    yield 'robustness', parent, 'robustness_probe', patch
            self.validation = iter(interleave(validate(p) for p in parents[:2]))
        return next(self.validation)

    def completed(self, phase):
        self.executed += 1
        self.minimized += int(phase == 'minimization')
