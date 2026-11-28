"""Order an initial proposal window on a small target/rival replay."""
from collections import Counter
from itertools import islice


def sample(db, scans, goals, rivals, maximum=8):
    eligible = {s['id']:s for s in scans}
    selected, labels = set(), {}
    for group in goals['groups']:
        for member in group['members']:
            observations = member['observations']
            best = min((r for r in observations if r['scan'] in eligible),
                       key=lambda r:(r.get('goal_role')!='reference',r['scan'] not in selected,r['id']),default=None)
            if best is None:
                return [], {}
            selected.add(best['scan'])
            if len(selected) > maximum:
                return [], {}
            for row in islice((r for r in observations if r['scan']==best['scan']),16):
                labels[row['id']] = (group['index'],member['identity_id'],row['scan'],row.get('cue',row['id']))
    # Include actual co-present competitors and explicitly measured outside rivals.
    from .database import encoded
    people = {m['identity_id'] for g in goals['groups'] for m in g['members']}
    from .body_identity import available
    sources = [('members','observations','')]
    if available(db):
        sources.append(('body_members','body_observations'," AND m.status='accepted'"))
    def records(where, args):
        clauses = [f'SELECT m.person,o.scan,o.id,o.cue FROM {m} m JOIN {o} o ON o.id=m.observation WHERE {where}{extra}'
                   for m,o,extra in sources]
        yield from db.execute(' UNION ALL '.join(clauses)+' ORDER BY 2,4,3', args*len(sources))
    outsider = False
    for person,scan,oid,_ in records("o.scan IN (SELECT value FROM json_each(?))",(encoded(sorted(selected)),)):
        if person not in people and oid not in labels:
            labels[oid] = -1
            outsider = True
            if sum(v==-1 for v in labels.values())>=16:
                break
    for rival in islice((r for r in rivals if r.get('rival') not in people),4):
        for _,scan,oid,_ in records('m.person=?',(rival['rival'],)):
            if scan in eligible and (scan in selected or len(selected)<maximum):
                labels[oid] = -1
                selected.add(scan)
                outsider = True
                break
    return ([s for s in scans if s['id'] in selected],labels) if outsider else ([],{})


def score(labels, predicted):
    groups = {}
    rivals = set()
    fragments = Counter()
    physical = {}
    for oid, label in labels.items():
        owner = predicted.get(oid)
        if owner is not None:
            slot = label if isinstance(label, tuple) else (label,None,None,oid)
            physical.setdefault(slot,set()).add(owner)
    conflicts = 0
    for (group,fragment,scan,cue), owners in physical.items():
        if len(owners) != 1:
            conflicts += 1
            continue
        owner, = owners
        if group < 0:
            rivals.add(owner)
        else:
            groups.setdefault(group,Counter())[owner] += 1
            if fragment is not None:
                fragments[group,fragment,owner] += 1
    contaminated = sum(n for counts in groups.values() for owner,n in counts.items()
                       if owner in rivals or any(owner in other for other in groups.values() if other is not counts))
    joined = sum(n*(n-1)//2 for counts in groups.values() for n in counts.values())
    joined -= sum(n*(n-1)//2 for n in fragments.values())
    return -(contaminated+conflicts),joined,sum(sum(v.values()) for v in groups.values())


class LocalScreen:
    def __init__(self, journal=None):
        self.journal = journal
        self.used = set()
        self.info = dict(replays=0,scenes=[],authority='proposal_order_only')

    def order(self, candidates, config, db, goals, scans, rivals, *, family='initial'):
        if family in self.used or len(scans)<32:
            yield from candidates
            return
        source = iter(candidates)
        batch = list(islice(source,4))
        if len(batch)<2:
            yield from batch
            yield from source
            return
        self.used.add(family)
        from .config import digest
        original = digest(batch)
        saved = None
        if self.journal:
            saved = self.journal.document.get('local_screens',{}).get(family)
            if not saved and len(self.used)==1:
                saved = self.journal.document.get('local_screen')
        if saved:
            if saved['input'] != original:
                raise ValueError('Incompatible local proposal screen in checkpoint.')
            self.info = saved['info']
            yield from saved['order']
            yield from source
            return
        local,labels = sample(db,scans,goals,rivals)
        if local:
            from .replay_simulation import prepared_history
            from .merge_report_policy import patched
            from .trial_evaluation import settings_for_replay, evaluate
            from .rebuild import rebuild_selected
            from .resource_errors import ResourceLimit
            from .hybrid_query_observer import observe_queries
            keep = {s['scene'] for s in local}
            args = goals['scope'],goals['namespace']
            excluded = [row[0] for row in db.execute('SELECT DISTINCT scene FROM scans WHERE scope=? AND namespace=?',args)
                        if row[0] not in keep]
            installation = db.execute("SELECT value FROM metadata WHERE key='installation'").fetchone()[0]
            self.info['scenes'] = sorted(keep)
            ranked = []
            try:
                with prepared_history(db,config,*args,installation,excluded_scenes=excluded) as seed, observe_queries(None):
                    for changes in batch:
                        try:
                            settings = settings_for_replay(patched(config,changes),None)
                        except ValueError:
                            continue
                        trial = evaluate('local-screen',settings,seed=seed,source=None,scope=args[0],namespace=args[1],
                                         installation=installation,scans=local,observations=labels,rebuild=rebuild_selected)
                        ranked.append((score(labels,trial['predictions']),changes))
                        self.info['replays'] += 1
            except ResourceLimit as error:
                self.info['limited'] = str(error)
            if len(ranked)==len(batch):
                ranked.sort(key=lambda item:item[0],reverse=True)
                batch = [patch for _,patch in ranked]
        if self.journal:
            entry = dict(input=original,order=batch,info=dict(self.info))
            self.journal.document.setdefault('local_screens',{})[family] = entry
            self.journal.events.append(dict(kind='local_family_screen',family=family,entry=entry))
        yield from batch
        yield from source
