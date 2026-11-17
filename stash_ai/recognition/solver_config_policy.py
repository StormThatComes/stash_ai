"""Constrain recommendations using matching policy and retained evidence."""
from collections import Counter

from .config import plain
from .config_check import Checks, DEFAULT_SCORE_GAP, report
from .config_check_matching import solver_constraints
from .merge_report_policy import parameter, patched
from .threshold_boundaries import at_or_above, at_or_below


class SolverChecks(Checks):
    """Evaluate only solver constraints without parsing YAML or rendering diagnostics."""
    def __init__(self, config):
        self.config = config
        self.score_gap = DEFAULT_SCORE_GAP
        self.issues = {}
        solver_constraints(self)

    def add(self, rule, paths, status, effect, suggestion, *, solver_constraint, **measures):
        if status != 'ok':
            self.issues[rule] = (paths, measures)


class StrictPolicy:
    def __init__(self, config):
        from .models import namespace
        from .scanning import scan_identity
        self.input = config
        self.portraits = plain(config.data['portraits'])
        self.namespace = namespace(config)
        self.scan_key = scan_identity(config, self.namespace)
        self.examples = {}
        self.input_issues = [r for r in report(config, solver_only=True)['checks'] if r['status'] != 'ok']
        self.rejected = Counter()
        self.protected = self.enabled_routes(config)

    @staticmethod
    def enabled_routes(config):
        from .body_identity import turn_feature
        from .fragment_endpoints import enabled as endpoints
        identity = config.data['sapiensid']['identities']
        active = {key for key in ('tracklet_matching', 'body_first_matching') if identity[key]}
        active.update(key for key in ('extended_track_graph', 'turn_recovery') if turn_feature(identity, key))
        if endpoints(identity):
            active.add('endpoint_matching')
        paths = {'sapiensid.identities.' + key for key in active}
        if config.data['clustering']['anchor_view_matching']:
            paths.add('clustering.anchor_view_matching')
        return paths

    def compatible(self, config):
        from .person_signal import same_contract
        from .scanning import scan_identity
        return scan_identity(config, self.namespace) == self.scan_key and same_contract(self.input, config)

    def violations(self, config, *, record=False, diagnostic=False):
        checks = SolverChecks(config)
        issues = set(checks.issues)
        if not diagnostic:
            issues.update(self.protected - self.enabled_routes(config))
        portraits = config.data['portraits']
        if portraits is not self.input.data['portraits'] and plain(portraits) != self.portraits:
            issues.add('portrait-settings-changed')
        if not self.compatible(config):
            issues.add('retained-scan-contract')
        if record:
            self.rejected.update(issues)
            for rule, (paths, measures) in checks.issues.items():
                self.examples.setdefault(rule, dict(settings={p: checks.value(p) for p in paths}, measures=measures))
            for rule in issues - checks.issues.keys():
                reason = {'retained-scan-contract': 'retained evidence incompatible',
                          'portrait-settings-changed': 'portrait settings are fixed during merge search'}.get(rule, 'input route disabled')
                self.examples.setdefault(rule, dict(reason=reason))
        return sorted(issues)

    def verify(self, config):
        issues = self.violations(config)
        if issues:
            raise ValueError('Strict solver config failed: ' + ', '.join(issues))

    def repairs(self, base, changes, *, initial=False):
        """Project the two endpoints onto their inclusive bounds, preserving inherited track settings."""
        config = patched(base, changes) if changes else base
        issues = self.violations(config)
        if not issues:
            return [changes]
        if issues != ['body-adjacent-cues']:
            self.violations(config, record=True)
            return []

        local = config.data['sapiensid']['identities']
        lower = 'cohesion_threshold' if local['track_similarity_threshold'] is None else 'track_similarity_threshold'
        track, match = local[lower], local['match_threshold']
        low_key, high_key = 'sapiensid.identities.' + lower, 'sapiensid.identities.match_threshold'
        gap = DEFAULT_SCORE_GAP
        # Each repair holds one endpoint as close as possible to its proposed value.
        # At the edges of [0, 1] both endpoints may have to move.
        proposals = [
            {low_key: max(0., at_or_below(match-gap)), high_key: max(match, gap)},
            {low_key: min(track, 1.-gap), high_key: min(1., at_or_above(track+gap))},
        ]
        repairs = []
        for proposal in proposals:
            edits = {k: v for k, v in proposal.items() if v != parameter(config.data, k)}
            candidate = changes | edits
            if not self.violations(patched(base, candidate)):
                cost = (sum(k in changes for k in edits), len(edits),
                        round(sum(abs(v-parameter(config.data, k)) for k, v in edits.items()), 6))
                repairs.append((cost, candidate))
        if not repairs:
            self.violations(config, record=True)
            return []
        repairs.sort(key=lambda item: item[0])
        return [p for _, p in repairs] if initial else [repairs[0][1]]

    def blocked(self, config):
        rows = [r for r in report(config, solver_only=True)['checks'] if r['status'] != 'ok']
        for row in rows:
            row['solver_parameters'] = list(row['settings'])
            row['repair_scope'] = 'Review the local match and effective adjacent-cue thresholds.'
        return rows
