"""Inspect final ownership and recorded decisions in private production replays."""
from collections import defaultdict
from functools import lru_cache
import json

from . import assignment_explanation
from .body_identity import available, enabled
from .clustering import matching_policy
from .database import encoded
from .identity_inspection import TABLES
from .identity_report import _assignment, _decisions, _merges
from .inspection_policy import selection
from .inspection_runtime import check_cancelled, stamp
from .merge_report_policy import patched
from .models import namespace
from .report_snapshot import snapshot
from .reference_roles import observation_sql


def sources(db, config, scope, ids):
    """Only current accepted observations belonging to the requested people."""
    rows = {}
    for kind, (table, members) in TABLES.items():
        if kind == 'body' and not (enabled(config) and available(db)):
            continue
        current, args = selection(config, kind)
        accepted = " AND m.status='accepted'" if kind == 'body' else ''
        for address, person, scene, cue in db.execute(
                f'SELECT o.id,m.person,s.scene,o.cue FROM {members} m JOIN {table} o ON o.id=m.observation '
                'JOIN scans s ON s.id=o.scan WHERE m.person IN (SELECT value FROM json_each(?)) '
                f'AND s.scope=? AND s.namespace=? AND {current}{accepted}', (encoded(ids), scope, namespace(config), *args)):
            rows[address] = dict(id=address, source=person, scene=scene, cue=cue, kind=kind)
    return rows


def decisions(db, settings, rows, scope):
    policy = matching_policy(settings)
    @lru_cache(maxsize=128)
    def family(person):
        members, links = _merges(db, dict(id=person, scope=scope, namespace=namespace(settings)))
        return members, {edge['source']: edge for edge in links}

    result = {}
    for key, row in rows.items():
        check_cancelled()
        if row['kind'] == 'face':
            owner = db.execute('SELECT person FROM members WHERE observation=?', (key,)).fetchone()
            if not owner:
                item = db.execute("SELECT status,json_extract(payload,'$.reason') AS reason FROM candidates c "
                                  f"WHERE {observation_sql('c')}=? AND policy=? ORDER BY rowid DESC LIMIT 1",
                                  (key,policy)).fetchone()
                if item:
                    result[key] = dict(role='unassigned', route=item['reason'] or item['status'], stage='',
                                       links=[], reference_sources=[])
                continue
            people, edges = family(owner[0])
            history = _decisions(db, dict(id=owner[0]), people, policy, observation=key).get(key, [])
            current = [h for h in history if h['matches_current_policy']]
            assignment = _assignment(current)
            record = assignment_explanation.face(current, assignment, edges, owner[0])
        else:
            item = db.execute("SELECT m.*,b.face FROM body_members m JOIN body_observations b ON b.id=m.observation "
                              "WHERE observation=?", (key,)).fetchone()
            if not item:
                continue
            from .body_identity import policy as body_policy
            if item['policy'] != body_policy(settings):
                continue
            proof = json.loads(item['proof'])
            if item['status'] != 'accepted':
                result[key] = dict(role='unassigned', route=proof.get('reason') or item['status'], stage='',
                                   links=[], reference_sources=[])
                continue
            _, edges = family(item['person'])
            record = assignment_explanation.body(proof, linked_face=item['face'], role=item['role'],
                status=item['status'], policy=item['policy'], current=True, merges=edges, target=item['person'])
        proofs = record['decisions']
        routes = sorted({p['reason'] for p in proofs if p.get('reason')})
        stages = sorted({p['stage'] for p in proofs if p.get('stage')})
        links = list(dict.fromkeys((edge['source'], edge['target'], edge['route']) for p in proofs for edge in p['merges']))
        refs = {r['observation'] for p in proofs for r in p['references']}
        result[key] = dict(role=record['role'], route=', '.join(routes), stage=', '.join(stages), links=links,
                           reference_sources=sorted({rows[ref]['source'] for ref in refs if ref in rows}))
    return result


def outcome(ids, rows, retained, predicted):
    dest = {p: set() for p in ids}
    counts = dict.fromkeys(ids, 0)
    lost = dict.fromkeys(ids, 0)
    for address, row in rows.items():
        src = row['source']
        counts[src] += 1
        owner = predicted.get(address)
        if owner is None:
            lost[src] += 1
        else:
            dest[src].add(owner)
    pair = dest[ids[0]] | dest[ids[1]]
    joined = len(pair) == 1 and all(counts[p] and not lost[p] for p in ids[:2])
    protected = None
    if len(ids) == 3:
        c = ids[2]
        protected = bool(counts[c] and not lost[c] and len(dest[c]) == 1 and dest[c].isdisjoint(pair))
    by_destination = defaultdict(set)
    for key, src in retained.items():
        if src is not None and src not in ids[:2] and predicted[key] in pair:
            by_destination[predicted[key]].add(src)
    outsiders = {p for people in by_destination.values() for p in people}
    return dict(joined=bool(joined), protected=protected, success=bool(joined and protected is not False),
                outsider_destinations=sorted(by_destination), counts=counts, lost=lost, outsiders=sorted(outsiders), destinations={p:sorted(v) for p,v in dest.items()})


def partial_state(source, owner, ids, outcome):
    target=ids[1] if source==ids[0] else ids[0]
    targets=outcome['destinations'][target]
    if owner is None:
        return 'unassigned'
    if len(targets)!=1:
        return 'target unresolved'
    if owner==targets[0]:
        return 'joined target'
    if owner in outcome['outsider_destinations']:
        return 'with another identity'
    return 'separate destination'


def attribution(ids, rows, before, after, result):
    if not result['joined']:
        return 'The pair did not fully join.', None
    for key, decision in after.items():
        row = rows[key]
        if row['source'] not in ids[:2]:
            continue
        for src, target, route in decision['links']:
            if {src, target} == set(ids[:2]):
                return (f"Post-formation consolidation · {route or 'recorded identity merge'} · "
                        f"supporting observation in scene {row['scene']}, cue {row['cue']}."), key
    for source, target in (ids[:2], ids[:2][::-1]):
        keys = [key for key,r in rows.items() if r['source']==source]
        seeds = [k for k in keys if before.get(k, {}).get('stage')=='seed'
                 or before.get(k, {}).get('route')=='body_scene_core']
        def attached(k):
            decision = after.get(k, {})
            return not decision.get('links') and (decision.get('role') in ('auxiliary','attachment') or
                decision.get('route') == 'candidate_group' and decision.get('stage') == 'match')
        if seeds and all(attached(k) for k in keys):
            example = next((k for k in seeds if target in after[k]['reference_sources']), None)
            if example:
                row = rows[example]
                return (f"Separate core avoided · {after[example]['route']} · former seed/core in scene "
                        f"{row['scene']}, cue {row['cue']} attached to the other identity's recorded evidence."), example
    return 'Final join verified. The retained decisions do not establish a decisive pair route or cue.', None


def _fresh(config, revision):
    if stamp(config.database) != revision:
        raise ValueError('The database changed. Refresh the scene and analyse the pair again before replay.')


def experiment(config, scope, ids, patches, revision, *, scenes=None, progress=None, refine_budget=None):
    from .inspection_merge_paths import _cancellable
    from .replay_simulation import replay
    from .inspection_pair_variants import patch_values, Refinement
    from .merge_report_policy import proposal
    if len(ids) not in (2,3) or len(set(ids)) != len(ids):
        raise ValueError('Select distinct identities for A, B and optional C.')
    _fresh(config, revision)
    from .config import digest
    names = {digest({}): 'baseline'}
    aliases, variants = {}, {}
    refinement = Refinement(config, patches['option'], refine_budget) if refine_budget is not None else None
    for name, patch in patches.items():
        if name == 'baseline':
            raise ValueError('The baseline name is reserved.')
        changes = patch_values(patch)
        signature = digest(changes)
        if signature not in names:
            variants[name] = patched(config, changes)
            names[signature] = name
        aliases[name] = names[signature]
    if refinement and aliases['option'] == 'baseline':
        raise ValueError('The selected option already uses the current YAML.')
    trials = {}
    with _cancellable(), snapshot(config, ids[0], scope=scope, other_identifiers=ids[1:]) as db:
        _fresh(config, revision)
        rows = sources(db, config, scope, ids)
        selected_scenes = set(scenes or ())
        if scenes is not None and any(row['scene'] not in selected_scenes for row in rows.values()):
            raise ValueError('Replay selection must include every current scene containing A, B and protected C.')
        totals = defaultdict(int)
        for kind, (_, table) in TABLES.items():
            if kind == 'body' and not (enabled(config) and available(db)):
                continue
            accepted = " AND status='accepted'" if kind == 'body' else ''
            for person, count in db.execute(f'SELECT person, COUNT(*) FROM {table} '
                    f'WHERE person IN (SELECT value FROM json_each(?)){accepted} GROUP BY person', (encoded(ids),)):
                totals[person] += count
        if {r['source'] for r in rows.values()} != set(ids):
            raise ValueError('A selected identity has no current observations in this selection.')
        def consume(name, observations, retained, predicted):
            return dict(pair=outcome(ids, rows, retained, predicted), owners={k:predicted.get(k) for k in rows})
        def collect(name, settings, item, inputs):
            records = item['decision_evidence']
            baseline = trials['baseline']['decisions'] if name != 'baseline' else records
            impact = item['versus_baseline']
            explanation, evidence = attribution(ids,rows,baseline,records,item['pair'])
            trial = dict(**item['pair'], owners=item['owners'], decisions=records, elapsed=item['elapsed_seconds'],
                         unassigned=impact['common_observations']-impact['assigned'], coverage_lost=impact['coverage_lost'],
                         attribution=explanation, evidence_id=evidence)
            from collections import Counter
            trial['partial'] = {p:Counter() for p in ids[:2]}
            for key, row in rows.items():
                if row['source'] in trial['partial']:
                    state = partial_state(row['source'],trial['owners'][key],ids,trial)
                    trial['partial'][row['source']][state] += 1
            if refinement:
                values = {} if name=='baseline' else patch_values(patches[name]) if name in patches else refinement.pending
                refinement.record(name, values, trial)
                # Keep details for the incumbent and latest probe until the generator makes its decision.
                for old in list(trials):
                    if old.startswith('probe-') and old != refinement.best:
                        # A previously successful patch may become relevant again after ablation.
                        # Re-run it if its full observation result is no longer available.
                        if trials[old]['success']:
                            refinement.forget(old)
                        del trials[old]
            trials[name] = trial
            # The serial inspector retains only its projection, independently of corpus size.
            for bulky in ('decision_evidence','owners','independent_person','hybrid'):
                item.pop(bulky, None)

        replayed = replay(config, variants, source=db, scope=scope, scenes=scenes, progress=progress, consume=consume,
                          inspect_trial=lambda trial, settings: decisions(trial, settings, rows, scope),
                          record_trial=collect, followup_variants=refinement.variants if refinement else None)
        _fresh(config, revision)
    if not trials:
        raise ValueError('No compatible scans were available.')
    result = dict(ids=ids, rows=rows, scenes=replayed['selection'], totals=dict(totals),
                  trials={'baseline':trials['baseline'], **{name:trials[actual] for name,actual in aliases.items()}})
    if refinement:
        result.update(minimized=proposal(config,refinement.values), probe_count=refinement.count,
                      budget_reached=not refinement.finished)
        result['trials']['minimized'] = trials[refinement.best]
    return result


def describe(result, name):
    item = result['trials'][name]
    baseline = result['trials']['baseline']
    lines = ['Replay on retained scans: ' + ('both identities fully joined.' if item['joined'] else 'pair did not fully join.'),
             'Current YAML baseline: ' + ('pair fully joined.' if baseline['joined'] else 'pair did not fully join.'),
             item['attribution']]
    if item['protected'] is not None:
        lines.append('Protected C fully assigned and separate: ' + ('yes' if item['protected'] else 'no'))
    for label, person in zip('ABC', result['ids']):
        outside = result['totals'][person] - item['counts'][person]
        lines.append(f"{label} · {person}: {item['counts'][person]} observations, {item['lost'][person]} unassigned, "
                     f"{outside} outside the current selection.")
    if not item['joined']:
        for label,person in zip('AB',result['ids']):
            lines.append(label+' → other side: '+', '.join(f'{count} {state}' for state,count in item['partial'][person].items()))
    if item['outsiders']:
        lines.append('Additional identities sharing a destination: ' + ', '.join(item['outsiders']))
    return '\n'.join(lines)
