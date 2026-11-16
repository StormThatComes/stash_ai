"""Cost estimates and replay-backed minimization for an inspector comparison."""
from contextlib import closing
from .body_identity import enabled, available
from .body_observations import contract
from .config import digest
from .database import encoded
from .inspection_runtime import check_cancelled, open_readonly
from .merge_report_policy import patched, parameter, proposal
from .merge_solver_domains import registry
from .merge_solver_tighten import MATCHING, tier
from .offline_evaluation import _selection


def preflight(config, scope, ids, entry, full, revision):
    from .inspection_pair_replay import sources, _fresh
    _fresh(config, revision)
    with closing(open_readonly(config.database)) as db:
        selected = None
        if not full:
            selected = {entry}
            selected.update(row['scene'] for row in sources(db,config,scope,ids).values())
        _, scans = _selection(db, config, selected, scope)
        scan_ids = encoded([s['id'] for s in scans])
        count_sql = ' WHERE scan IN (SELECT value FROM json_each(?))'
        faces = db.execute('SELECT COUNT(*) FROM observations'+count_sql,(scan_ids,)).fetchone()[0]
        bodies = 0
        if enabled(config) and available(db):
            bodies = db.execute('SELECT COUNT(*) FROM body_observations'+count_sql+' AND contract=?',
                                (scan_ids,contract(config))).fetchone()[0]
    return dict(scenes=[s['scene'] for s in scans], faces=faces, bodies=bodies, full=full)


def patch_values(patch):
    return {c['parameter']:c['proposed'] for c in patch['changes']}


def admissible(candidate, original):
    return (candidate['success'] and set(candidate['outsiders']).issubset(original['outsiders'])
            and candidate['unassigned'] <= original['unassigned'] and candidate['coverage_lost'] <= original['coverage_lost'])


class Refinement:
    """One incumbent; every accepted change has a completed production trial."""
    def __init__(self, config, initial, budget):
        if type(budget) is not int or budget < 1:
            raise ValueError('The refinement replay budget must be a positive integer.')
        self.config, self.budget = config, budget
        self.values = patch_values(initial)
        self.known, self.metrics = {}, {}
        self.count = 0
        self.best = 'option'
        self.finished = False

    def record(self, name, values, outcome):
        self.known[digest(values)] = name
        self.metrics[name] = {k:outcome[k] for k in ('success','outsiders','unassigned','coverage_lost')}

    def forget(self, name):
        for signature in [s for s,trial in self.known.items() if trial == name]:
            del self.known[signature]

    def try_values(self, values):
        check_cancelled()
        name = self.known.get(digest(values))
        if name is None:
            if self.count >= self.budget:
                return False
            try:
                config = patched(self.config, values)
            except ValueError:
                return False
            self.count += 1
            name = f'probe-{self.count}'
            self.pending = values
            yield name, config
        if admissible(self.metrics[name], self.metrics['option']):
            self.values, self.best = values, name
            return True
        return False

    def remove(self):
        while self.values and self.count < self.budget:
            keys = sorted(self.values)
            groups = [keys, *[(k,) for k in keys]]
            groups.extend([k for k in keys if k.startswith(prefix)] for prefix in ('recognition.','clustering.','sapiensid.'))
            for match, cohesion in (('recognition.match_threshold','clustering.cohesion_threshold'),
                                   ('sapiensid.identities.match_threshold','sapiensid.identities.cohesion_threshold'),
                                   ('sapiensid.global_matching.match_threshold','sapiensid.global_matching.cohesion_threshold')):
                if match in self.values and cohesion in self.values:
                    groups.append((match,cohesion))
            for group in groups:
                if group and (yield from self.try_values({k:v for k,v in self.values.items() if k not in group})):
                    break
            else:
                return

    def variants(self):
        if not self.metrics['option']['success']:
            raise ValueError('Minimization requires an option that meets every selected goal.')
        yield from self.remove()
        domains = registry(self.config)
        # Probe each changed matching gate before narrowing a single boundary.
        # Admission gates may return toward the loaded policy, without discarding more observations.
        keys = sorted((k for k in self.values if k in domains and domains[k].kind is not bool
                       and type(self.values[k]) in (int,float)),key=lambda k:(tier(k),k))
        stopped = {}
        while keys and self.count < self.budget:
            advanced = False
            for key in list(keys):
                if key not in self.values:
                    keys.remove(key)
                    continue
                domain, start = domains[key], self.values[key]
                context = digest({k:v for k,v in self.values.items() if k != key})
                bound = stopped.get((context,key))
                end = domain.low if domain.relax > 0 else domain.high
                if key not in MATCHING:
                    end = parameter(self.config.data,key)
                if end is None or (end-start)*domain.relax >= 0:
                    keys.remove(key)
                    continue
                target = domain.project(end if bound is None else (start+bound)/2)
                if target == start or target == bound:
                    keys.remove(key)
                    continue
                candidate = dict(self.values, **{key:target})
                candidate = patch_values(proposal(self.config,candidate))
                if not (yield from self.try_values(candidate)):
                    stopped[context,key] = target
                advanced = True
                if self.count >= self.budget:
                    break
            if not advanced:
                break
        yield from self.remove()
        self.finished = self.count < self.budget
