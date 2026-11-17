"""Read-only diagnostics for the effective recognition policy."""
import json
import sys

from .config import parse_yaml, plain
from .config_layout import POINT_FIELDS, POINT_STAGES, PUBLIC_PATHS, get

DEFAULT_SCORE_GAP = .02


class Checks:
    def __init__(self, config, score_gap):
        self.config = config
        self.score_gap = score_gap
        self.rows = []
        self.pose_source = (parse_yaml(config.source).get('pose', {})
                            if config.document_version == 2 and config.source else {})

    def value(self, path):
        return get(self.config.data, PUBLIC_PATHS[path])

    def add(self, rule, paths, status, effect, suggestion, *, solver_constraint=False, preserve_input=False, **measures):
        values = {}
        for path in paths:
            if self.config.document_version == 1:
                values[PUBLIC_PATHS[path]] = plain(self.value(path))
                continue
            values[path] = plain(self.value(path))
            parts = path.split('.')
            if (len(parts) == 3 and parts[0] == 'pose' and parts[1] in POINT_STAGES
                    and parts[2] in POINT_FIELDS and self.pose_source.get(parts[1], {}).get(parts[2]) is None):
                values['pose.point_quality.' + parts[2]] = values[path]
                values[path] = None
        self.rows.append(dict(rule=rule, status=status, solver_constraint=solver_constraint or preserve_input,
                              preserve_input=preserve_input, settings=values, measures=measures,
                              effect=effect, suggestion=suggestion if status != 'ok' else ''))

    def detail(self, rule, paths, effect, **measures):
        """Describe effective settings for --all, including valid equal cutoffs."""
        self.add(rule, paths, 'ok', effect, '', **measures)

    def interval(self, rule, paths, upper, lower, effect):
        self.detail(rule, paths, effect, lower=lower, upper=upper, gap=upper-lower)

    def effective_max(self, rule, paths, values, effect):
        self.detail(rule, paths, effect, effective=max(values))

    def covers(self, rule, paths, available, required, effect, suggestion):
        self.add(rule, paths, 'warning' if available < required else 'ok', effect, suggestion,
                 available=available, required=required)

    def route(self, rule, paths, reachable, effect, suggestion, *, enabled=True, **measures):
        """Strict proposals preserve this route when the input can use it."""
        if enabled:
            self.add(rule, paths, 'ok' if reachable else 'warning', effect, suggestion,
                     preserve_input=True, **measures)

    def scene_capacity(self, rule, paths, required, description):
        cap = 'execution.scan.max_cues'
        available = self.value(cap)
        if available:
            self.route(rule, (cap, *paths), available >= required,
                       f'{description} needs at least {required} sampled cues in one scene.',
                       'Increase the scan cue limit or review the independent support requirement.',
                       available=available, required=required)


def run_rules(checks, *, solver_only=False):
    from .config_check_matching import matching, solver_constraints
    from .config_check_quality import quality
    if not solver_only:
        quality(checks)
        matching(checks)
    solver_constraints(checks)


def report(config, score_gap=DEFAULT_SCORE_GAP, *, solver_only=False):
    if not 0 <= score_gap <= 1:
        raise ValueError('--min-score-gap must be in [0, 1].')
    checks = Checks(config, score_gap)
    run_rules(checks, solver_only=solver_only)
    counts = {status: sum(r['status'] == status for r in checks.rows)
              for status in ('warning', 'advice', 'ok')}
    return dict(config=str(config.source_path) if config.source_path else 'built-in defaults',
                version=config.document_version, minimum_score_gap=score_gap, counts=counts,
                checks=checks.rows)


def display(result, *, show_all=False, as_json=False, stream=None):
    stream = sys.stdout if stream is None else stream
    rows = [r for r in result['checks'] if show_all or r['status'] != 'ok']
    if as_json:
        print(json.dumps(dict(result, checks=rows), indent=2, allow_nan=False), file=stream)
        return
    counts = result['counts']
    print(f"Recognition config check ({result['config']}, v{result['version']}): "
          f"warnings={counts['warning']}, tuning notes={counts['advice']}, passed={counts['ok']}.", file=stream)
    for row in rows:
        scope = ' [strict solver]' if row['solver_constraint'] else ''
        if row['preserve_input']:
            scope = ' [strict solver: preserve input route]'
        print(f"[{row['status'].upper()}]{scope} {row['rule']}  {row['effect']}", file=stream)
        print('  ' + ', '.join(f'{p}={json.dumps(v)}' for p, v in row['settings'].items()), file=stream)
        if row['measures']:
            fields = (f'{key}={format(value, ".8g") if isinstance(value, float) else value}'
                      for key, value in row['measures'].items())
            print('  ' + ', '.join(fields), file=stream)
        if row['suggestion']:
            print('  ' + row['suggestion'], file=stream)
