"""Bounded coordinator-owned progress for interactive terminals and CI logs."""
from collections import Counter, deque
from statistics import median
from threading import RLock
from time import monotonic

from .console_layout import compact, clock
from .progress import status_lines
from .live_resources import gib


def worker_row(snapshot, now, width):
    resources = snapshot.get('resources', {})
    reported = snapshot.get('reported', snapshot['updated'])
    completed = snapshot.get('last_completed')
    age = now - reported
    if age >= 15:
        state = 'no telemetry ' + clock(age)
    elif completed is not None and now - completed >= 15:
        state = 'last progress ' + clock(now - completed) + ' ago'
    else:
        state = clock(now - snapshot['started'])
    detail = [state]
    if snapshot.get('scene') is not None:
        detail.append('scene ' + str(snapshot['scene']))
    for work in snapshot.get('work', ()):
        total = '?' if work['total'] is None else work['total']
        detail.append(f"{work['completed']}/{total} {work['unit']}")
    stage, = compact([snapshot['stage']], max(8, min(34, width-len(' · '.join(detail))-3)))
    if resources.get('rss') is not None:
        detail.append('RSS ' + gib(resources['rss']))
    if resources.get('cores') is not None:
        detail.append(f"CPU {resources['cores']:.1f}")
    if snapshot.get('dropped'):
        detail.append(f"telemetry skipped {snapshot['dropped']}")
    return stage + ' · ' + ' · '.join(detail)


class SolverDisplay:
    def __init__(self, console, maximum, slots):
        self.console, self.maximum, self.slots = console, maximum, slots
        self.lock = RLock()
        self.rows = {}
        self.fresh = self.restored = self.failed = 0
        self.best, self.best_state = None, {}
        self.first_solution = None
        self.durations = deque(maxlen=31)
        self.pool, self.preparing = '', ''
        self.last_initialization = None
        self.search_phase = None
        self.process_samples = {}
        self.failure_groups = Counter()
        self.applying = None
        console.status_layout = self.layout
        console.heartbeat_layout = self.heartbeat

    def header(self):
        return (f'Solver {self.fresh + self.restored}/{self.maximum} · fresh {self.fresh} · '
                f'restored {self.restored} · failed {self.failed} · best {self.best or "pending"}')

    def layout(self, row, fields, width):
        if row['phase'] != 'Solve merge goals':
            lines = status_lines(row, fields, width)
            return compact([*lines, fields['validation']], width) if fields.get('validation') else lines
        with self.lock:
            now = monotonic()
            lines = [self.header(), f"Total {clock(row['total_elapsed'])} · phase {clock(row['elapsed'])}"]
            if self.durations:
                lines[-1] += ' · median successful replay ' + clock(median(self.durations))
            if self.pool or self.preparing:
                lines.append(self.pool or self.preparing)
            if self.search_phase:
                lines.append('Search: ' + self.search_phase)
            for name, value in self.rows.items():
                text = worker_row(value, now, width-12) if isinstance(value, dict) else value
                lines.append(f'{name:<12}{text}')
            if self.process_samples:
                from .live_resources import sampled_processes
                lines.append('Workers: ' + sampled_processes(self.process_samples.values(), now))
            if fields.get('process'):
                lines.append('Coordinator: ' + fields['process'])
            plans = {v['resources']['trial_budget'] for v in self.rows.values() if isinstance(v, dict)
                     and v.get('resources', {}).get('trial_budget')}
            if plans:
                lines.append('Admission per trial: ' + ', '.join(gib(plan) for plan in sorted(plans)))
            if fields.get('disk'):
                lines.append('SSD: ' + fields['disk'])
            leases = [v['resources'] for v in self.rows.values() if isinstance(v, dict)
                      and 'scratch_reserved' in v.get('resources', {})]
            if leases:
                free = [r['scratch_free'] for r in leases if r.get('scratch_free') is not None]
                lines.append('Trial scratch held: ' + gib(sum(r['scratch_reserved'] for r in leases))
                             + ' · min sampled volume free ' + gib(min(free) if free else None))
            return compact(lines, width)

    def heartbeat(self, row, fields):
        if row['phase'] != 'Solve merge goals':
            return ' · '.join(str(v) for v in (row['phase'], clock(row['elapsed']), fields.get('validation'), fields.get('stage')) if v)
        with self.lock:
            workers = [(name, value) for name, value in self.rows.items() if isinstance(value, dict)]
            detail = ''
            if workers:
                name, value = min(workers, key=lambda entry: entry[1]['started'])
                detail = f' · {name} {worker_row(value, monotonic(), 70)}'
            if self.applying:
                detail += f' · {self.applying[0]} {self.applying[1]}'
            phase = ' · ' + self.search_phase if self.search_phase else ''
            return self.header() + phase + ' · ' + (self.pool or self.preparing) + detail

    show_winner = True

    def __call__(self, name, stage, item):
        with self.lock:
            message = self.update(name, stage, item or {})
        if message:
            self.console.message(message, decorative=True)

    def update(self, name, stage, item):
        if stage == 'partial':
            return f"Partial {item['kind']} diagnostics · {item['reason']} · {str(item.get('detail', ''))[:160]}"
        if stage == 'plan':
            return 'Merge solver execution: plan ' + item['text']
        if name == 'inputs':
            return 'Merge solver inputs: ' + stage
        if name == 'initialization':
            label = item.get('stage', stage)
            self.preparing = label
            if label != self.last_initialization:
                self.last_initialization = label
                return 'Merge solver initialization: ' + label
        elif stage == 'start':
            self.preparing = ''
            self.search_phase = item.get('search_phase', 'baseline' if name == 'baseline' else self.search_phase)
            self.rows[name] = 'queued / starting'
            if name == 'baseline':
                self.pool = f'baseline runs alone · {self.slots} candidate slots after baseline'
            else:
                self.pool = self.pool if self.pool.startswith('Pool ') else f'Candidate slots {self.slots}'
        elif stage == 'pool':
            self.pool = item.get('compact', item['text'])
            self.process_samples = {pid: dict(row, sampled=row.get('sampled', observed))
                                    for pid, (observed, row) in item.get('process_samples', {}).items()}
        elif stage == 'coordinator':
            self.applying = (name, item['text'])
            self.rows[name] = item['text']
        elif stage in ('ready', 'status'):
            self.rows[name] = item.get('snapshot') or item.get('compact', item['text'])
        elif stage.startswith('stage: '):
            if name in self.rows:
                self.rows[name] = item
        elif stage in ('failed', 'complete'):
            if self.applying and self.applying[0] == name:
                self.applying = None
            self.rows.pop(name, None)
            restored = item.get('restored_result', False)
            self.restored += int(restored)
            self.fresh += int(not restored)
            self.failed += int(stage == 'failed')
            if not restored and stage == 'complete':
                self.durations.append(item['elapsed_seconds'])
            if stage == 'failed':
                key = (item['stage'], item['reason'], (item.get('resource') or {}).get('budget'), restored)
                self.failure_groups[key] += 1
                if not restored and self.failure_groups[key] == 1:
                    return f"Merge solver {name} failed · {item['stage']} · {item['reason']}: {item['error']}"
                return None
            best = item['best_trial']
            state = item['winner_state']
            if self.best != best:
                changed = [f'{key} {self.best_state[key]} → {value}' for key, value in state.items()
                           if key != 'strength' and key in self.best_state and self.best_state[key] != value]
                if 'strength' in self.best_state:
                    for index, (before, after) in enumerate(zip(self.best_state['strength'], state['strength'])):
                        if before != after:
                            changed.append(f'matching strength tier {index + 1}: {before:g} → {after:g}')
                            break
                self.best, self.best_state = best, state
                detail = ', '.join(changed[:3]) if changed else f"merged sets {state['sets']}"
                prefix = 'best'
                if state['solved'] and self.first_solution is None:
                    self.first_solution = best
                    prefix = 'first feasible'
                if not restored or self.restored == 1:
                    from .solver_explanation import parameter_change
                    delta = item.get('winning_delta', ())
                    if delta:
                        detail += ' · ' + parameter_change(delta[0])
                        if len(delta) > 1:
                            detail += f' (+{len(delta)-1} more)'
                    evidence = item.get('blockers', ())
                    if evidence:
                        detail += '\n  ' + ' | '.join(evidence[:3])
                        if len(evidence) > 3:
                            detail += f' (+{len(evidence)-3} sets)'
                    phase = item.get('winning_phase', 'baseline')
                    return f'Merge solver {prefix} {best} · {phase} · {detail}'
        return None

    def finish(self, result):
        search = result.get('search', {})
        best = result.get('best_result')
        if best:
            phrase = f"best completed trial: {best['trial']} ({best['quality']})"
            phrase += ' · configuration=' + ('changed' if best['changed'] else 'unchanged')
        else:
            phrase = 'no completed trial'
        trials = result.get('simulation', {}).get('trials', {})
        selected = result.get('selected_trial')
        groups = trials.get(selected, {}).get('goal_result', {}).get('groups', [])
        merged = sum(g.get('identity_fully_merged', g.get('fully_merged', False)) for g in groups)
        assigned = sum(g.get('assigned', 0) for g in groups)
        attempted = search.get('attempted', 0)
        restored = search.get('restored_trials', 0) + search.get('recovered_evaluations', 0)
        line = (f'Merge solver {phrase} · stop={search.get("stop_reason", "unknown")}\n'
                f'  merged sets {merged}/{len(groups)} · assigned={assigned}'
                f' · Solver {attempted}/{search.get("max_trials", self.maximum)} · fresh {attempted-restored}'
                f' · restored {restored} · failed {search.get("failed_trials", 0)}'
                )
        if best and (coverage := best.get('assignment_coverage')):
            from .merge_solver_coverage import summary
            line += '\n  ' + summary(coverage)
        if best and best['unmet_conditions']:
            from collections import Counter
            reasons = Counter()
            for condition in best['unmet_conditions']:
                reasons[condition['reason']] += condition.get('count', 1)
            line += ' · unmet: ' + ', '.join(f'{key} ({count})' for key, count in reasons.most_common(3))
        if self.failure_groups:
            reasons = []
            for (stage, reason, budget, restored), count in self.failure_groups.most_common(3):
                reasons.append(f"{stage}/{budget or reason} ×{count} ({'restored' if restored else 'fresh'})")
            line += '\n  failed trials: ' + ', '.join(reasons)
        return line
