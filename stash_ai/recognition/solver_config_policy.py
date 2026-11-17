"""Project retained-evidence proposals into the region accepted by check-config."""
from collections import Counter
from dataclasses import replace
from heapq import heappop, heappush
from itertools import count
import math

from .config_check import Checks, DEFAULT_SCORE_GAP, report, run_rules
from .config_layout import PUBLIC_PATHS, internal_path, public_path
from .merge_report_policy import parameter, patched
from .merge_solver_domains import Domain, registry


class ActiveChecks(Checks):
    """Run the same rules without parsing source YAML or rendering passing rows."""
    def __init__(self, config):
        self.config = config
        self.score_gap = DEFAULT_SCORE_GAP
        self.active = set()
        self.issues = {}
        run_rules(self)

    def add(self, rule, paths, status, effect, suggestion, **measures):
        self.active.add(rule)
        if status != 'ok':
            self.issues[rule] = (paths, measures)


def shortfall(measures, step):
    if 'gap' in measures:
        return max(step, measures['minimum_gap'] - measures['gap'])
    if 'required' in measures:
        return max(step, measures['required'] - measures['available'])
    return step


def changed_data(config, changes):
    # Only the touched dictionaries are writable. Solver threads never share mutations.
    data = dict(config.data)
    for path, value in changes.items():
        source, target = config.data, data
        keys = path.split('.')
        for key in keys[:-1]:
            if target[key] is source[key]:
                target[key] = dict(source[key])
            source, target = source[key], target[key]
        target[keys[-1]] = value
    return replace(config, data=data)


class StrictPolicy:
    def __init__(self, config):
        from .models import namespace
        from .scanning import scan_identity
        self.input = config
        self.namespace = namespace(config)
        self.scan_key = scan_identity(config, self.namespace)
        self.required = ActiveChecks(config).active
        self.domains = registry(config)
        self.domains['sapiensid.identities.tracklet_matching'] = Domain(bool)
        self.portrait_domains = {'portraits.' + key: Domain(high=upper) for key, upper in (
            ('min_face_pixels', 4096), ('min_sharpness', 100000), ('min_visible_face_fraction', 1),
            ('max_pose_deviation', 1), ('max_clipped_fraction', 1))}
        self.portrait_domains.update({'portraits.weights.' + k: Domain(high=100)
                                     for k in config.data['portraits']['weights']})
        self.domains.update(self.portrait_domains)
        self.examples = {}
        self.input_issues = [r for r in report(config)['checks'] if r['status'] != 'ok']
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

    def check(self, config):
        state = ActiveChecks(config)
        missing = self.required - state.active
        missing.update(self.protected - self.enabled_routes(config))
        return state, missing

    def reject(self, state, missing):
        self.rejected.update(state.issues.keys() | missing)
        for rule, (paths, measures) in state.issues.items():
            if rule not in self.examples:
                self.examples[rule] = dict(settings={p: state.value(p) for p in paths}, measures=measures)
        for rule in missing:
            self.examples.setdefault(rule, dict(reason='retained evidence incompatible' if rule == 'retained-scan-contract' else 'input route disabled'))

    def moves(self, config, rule, paths, measures):
        if rule == 'body-track-features':
            yield 'sapiensid.identities.tracklet_matching', True
            return
        for public in paths:
            key = PUBLIC_PATHS[public]
            if rule.startswith('portrait-') and key not in self.portrait_domains:
                continue
            domain = self.domains.get(key)
            if domain is None or domain.kind is bool:
                continue
            value = parameter(config.data, key)
            integral = domain.kind is int or key.endswith('face_pixels')
            step = 1 if integral else DEFAULT_SCORE_GAP
            if value is None:
                if rule != 'face-auxiliary-quality-band' or not key.startswith('clustering.min_anchor_'):
                    continue
                suffix = key.removeprefix('clustering.min_anchor_')
                floor = max(config.data['detection']['min_' + suffix],
                            config.data['clustering']['min_auxiliary_' + suffix] or 0)
                yield key, round(floor + step, 6)
                continue
            delta = shortfall(measures, step if 'minimum_gap' not in measures or not measures['minimum_gap'] else .000001)
            if 'admission_limit' in measures:
                delta = abs(measures['target'] - measures['admission_limit']) + step
            if integral:
                delta = max(1, math.ceil(delta - 1e-10))
            for change in (delta, -delta):
                result = round(min(domain.high, max(domain.low, value + change)), 6)
                if integral:
                    result = int(round(result))
                if result != value:
                    yield key, result

    def repairs(self, base, changes, *, initial=False):
        """Try both endpoints of broken relations. Bound static work separately from replays."""
        start = changed_data(base, changes)
        state, missing = self.check(start)
        if not state.issues and not missing:
            if self.compatible(start):
                return [changes]
            self.rejected['retained-scan-contract'] += 1
            return []
        if missing:
            self.reject(state, missing)
            return []
        order = count()
        queue = []
        seen = set()
        solutions = []
        def enqueue(patch, settings, checks):
            key = tuple(sorted(patch.items()))
            if key in seen:
                return
            seen.add(key)
            # Prefer to preserve the proposed coordinates, then minimize the repair.
            edits = [(k, v) for k, v in patch.items() if changes.get(k, parameter(base.data, k)) != v]
            distance = sum(abs(v - (parameter(start.data, k) or 0)) for k, v in edits)
            cost = (len(checks.issues), sum(shortfall(m, .000001) for _, m in checks.issues.values()),
                    sum(k in changes for k, _ in edits), len(edits), round(distance, 6))
            heappush(queue, (cost, next(order), patch, settings, checks))
        enqueue(changes, start, state)
        for _ in range(160 if initial else 16):
            if not queue:
                break
            _, _, patch, settings, checks = heappop(queue)
            if not checks.issues:
                try:
                    patched(base, patch)
                except ValueError:
                    continue
                if not self.compatible(settings):
                    self.rejected['retained-scan-contract'] += 1
                    continue
                solutions.append(patch)
                if not initial or len(solutions) == 3:
                    break
                continue
            # Fix one relation at a time. Other failing relations stay visible and
            # compound max/min gates may require several changes to their operands.
            rule, (paths, measures) = next(iter(checks.issues.items()))
            for key, value in self.moves(settings, rule, paths, measures):
                # Keep non-search dimensions and newly enabled gates fixed after preflight.
                if not initial and (key in self.portrait_domains or parameter(base.data, key) is None):
                    continue
                child = dict(patch, **{key: value})
                if tuple(sorted(child.items())) in seen:
                    continue
                candidate = changed_data(settings, {key: value})
                updated, missing = self.check(candidate)
                if missing:
                    continue
                remaining = updated.issues.get(rule)
                if remaining and shortfall(remaining[1], .000001) > shortfall(measures, .000001) + 1e-10:
                    continue
                enqueue(child, candidate, updated)
        if not solutions:
            self.reject(state, set())
        return solutions

    def violations(self, config, *, record=False):
        state, missing = self.check(config)
        if not self.compatible(config):
            missing.add('retained-scan-contract')
        if record:
            self.reject(state, missing)
        return sorted(state.issues.keys() | missing)

    def verify(self, config):
        issues = self.violations(config)
        if issues:
            raise ValueError('Strict solver config failed: ' + ', '.join(issues))

    def blocked(self, config):
        rows = [r for r in report(config)['checks'] if r['status'] != 'ok']
        for row in rows:
            paths = [internal_path(k) for k in row['settings']]
            row['solver_parameters'] = [public_path(k) for k in paths if k in self.domains]
            canonical = row['settings'] if config.document_version == 2 else (public_path(k) for k in paths)
            if not row['solver_parameters'] and any(k.startswith(('pose.', 'face.quality.', 'body.collection.'))
                                                    for k in canonical):
                row['repair_scope'] = 'scan settings; review YAML and recollect affected scenes'
            else:
                row['repair_scope'] = 'review YAML; bounded retained-evidence repair found no compliant configuration'
        return rows
