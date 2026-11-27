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
        signals = sorted({p['metrics']['signal'] for p in proofs if p['metrics'].get('signal')})
        selected_refs = sorted(refs.intersection(rows))
        result[key] = dict(role=record['role'], route=', '.join(routes), stage=', '.join(stages), links=links,
                           signal=', '.join(signals), references=selected_refs,
                           reference_sources=sorted({rows[ref]['source'] for ref in selected_refs}))
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


def replay_controls(patches):
    return {person for patch in patches.values() for person in patch.get('controls', ())}


def control_results(rows, predicted, destinations):
    """A protected rival must retain every sample at one distinct destination."""
    owners = defaultdict(set)
    for key, row in rows.items():
        owners[row['source']].add(predicted.get(key))
    return {p: len(values) == 1 and None not in values and not values.intersection(destinations)
            for p, values in owners.items()}


def apply_controls(trial, patch):
    if not patch.get('controls') and not patch.get('watch') and not trial.get('watched'):
        return trial
    checks = {p: trial['control_results'].get(p) for p in patch.get('controls', ())}
    watched = {k:trial['watched'].get(k) for k in patch.get('watch', ())}
    return dict(trial, controls=checks, watched=watched,
                success=trial['joined'] and trial['protected'] is not False and all(v is True for v in checks.values()))


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


def _mark_private_anchors(target, omitted):
    # Replay shares its immutable embeddings through a TEMP view. Overlay only
    # the quality projection and keep rowids, indexes and the shared seed intact.
    if target.execute("SELECT 1 FROM temp.sqlite_master WHERE type='view' AND name='observations'").fetchone():
        columns = [r[1] for r in target.execute('PRAGMA retained.table_info(observations)')]
        target.execute('CREATE TEMP TABLE omitted_face_anchors(id TEXT PRIMARY KEY)')
        target.executemany('INSERT INTO omitted_face_anchors VALUES (?)', [(k,) for k in omitted])
        fields = ["CASE WHEN id IN omitted_face_anchors THEN json_set(quality,'$._inspection_auxiliary_only',1) "
                  "ELSE quality END AS quality" if c == 'quality' else c for c in columns]
        from .snapshot_vectors import projection
        target.execute('DROP VIEW temp.observations')
        target.execute('CREATE TEMP VIEW observations AS SELECT rowid AS rowid,' + ','.join(fields) +
                       ' FROM ('+projection(target,'retained','observations')+')')
    else:
        target.execute("UPDATE observations SET quality=json_set(quality,'$._inspection_auxiliary_only',1) "
                       "WHERE id IN (SELECT value FROM json_each(?))", (encoded(omitted),))


def experiment(config, scope, ids, patches, revision, *, scenes=None, progress=None, refine_budget=None, omitted_stage=None):
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
    visits = defaultdict(dict)
    if omitted_stage is not None:
        from .replay_intervention import STAGES, members, trial
        if omitted_stage not in STAGES or set(patches)!={'enabled','omitted'} or patches['enabled']!=patches['omitted']:
            raise ValueError('The stage check requires two identical patches and one isolated stage.')
    for name, patch in patches.items():
        if name == 'baseline':
            raise ValueError('The baseline name is reserved.')
        changes = patch_values(patch)
        diagnostic = bool(patch.get('omit_face_anchors') or patch.get('omit_body_anchors'))
        experiment_key = [changes, patch.get('omit_face_anchors',()), patch.get('omit_body_anchors',())]
        if omitted_stage and name == 'omitted':
            experiment_key.append(omitted_stage)
        signature = digest(experiment_key) if diagnostic or (omitted_stage and name=='omitted') else digest(changes)
        if signature not in names:
            variants[name] = patched(config, changes)
            names[signature] = name
        aliases[name] = names[signature]
    if refinement and (patches['option'].get('omit_face_anchors') or patches['option'].get('omit_body_anchors')):
        raise ValueError('An anchor ablation is a diagnostic experiment, not a YAML patch.')
    if refinement and aliases['option'] == 'baseline':
        raise ValueError('The selected option already uses the current YAML.')
    trials = {}
    with _cancellable(), snapshot(config, ids[0], scope=scope, other_identifiers=ids[1:]) as db:
        _fresh(config, revision)
        rows = sources(db, config, scope, ids)
        controls = replay_controls(patches) - set(ids[:2])
        control_rows = sources(db, config, scope, sorted(controls)) if controls else {}
        watched = {k for patch in patches.values() for k in patch.get('watch', ())}
        selected_scenes = set(scenes or ())
        if scenes is not None and any(row['scene'] not in selected_scenes for bank in (rows, control_rows) for row in bank.values()):
            raise ValueError('Replay selection must include every current scene containing A, B and protected controls.')
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
            pair = outcome(ids, rows, retained, predicted)
            destinations = set(pair['destinations'][ids[0]]) | set(pair['destinations'][ids[1]])
            pair['control_results'] = control_results(control_rows,predicted,destinations)
            pair['watched'] = {}
            for key in watched:
                owner = predicted.get(key)
                sides = [label for label,p in zip('AB',ids) if owner in pair['destinations'][p]]
                state = 'unassigned' if owner is None else '+'.join(sides) if sides else 'other identity'
                pair['watched'][key] = state
            return dict(pair=pair, owners={k:predicted.get(k) for k in rows})
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
                refinement.record(name, values, apply_controls(trial, patches['option']))
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

        def context(name):
            return trial(omitted_stage if name=='omitted' else None,visits[name])
        def prepare_private(name, target, settings):
            option = patches.get(name,{})
            bodies = option.get('omit_body_anchors',())
            if bodies:
                if any(k not in rows or rows[k]['kind'] != 'body' or rows[k]['source'] not in ids[:2] for k in bodies):
                    raise ValueError('Excluded body anchors must belong to the compared pair.')
                settings.data['sapiensid']['identities']['_inspection_omit_body_anchors'] = list(bodies)
            omitted = option.get('omit_face_anchors',())
            if omitted:
                if any(k not in rows or rows[k]['kind'] != 'face' or rows[k]['source'] not in ids[:2] for k in omitted):
                    raise ValueError('Excluded anchors must belong to the compared pair.')
                # This trial owns both the input rows and settings. Ordinary query
                # eligibility and measurements remain available to auxiliary matching.
                settings.data['clustering']['_inspection_omit_anchors'] = list(omitted)
                _mark_private_anchors(target, omitted)
        private = any(p.get('omit_face_anchors') or p.get('omit_body_anchors') for p in patches.values())
        replayed = replay(config, variants, source=db, scope=scope, scenes=scenes, progress=progress, consume=consume,
                          inspect_trial=lambda trial, settings: decisions(trial, settings, rows, scope),
                          record_trial=collect, prepare_private_trial=prepare_private if private else None,
                          trial_context=context if omitted_stage else None, followup_variants=refinement.variants if refinement else None)
        _fresh(config, revision)
    if not trials:
        raise ValueError('No compatible scans were available.')
    projected = {}
    def projection(name, actual):
        patch = patches[name]
        key = actual, tuple(sorted(patch.get('controls',()))), tuple(sorted(patch.get('watch',())))
        if key not in projected:
            projected[key] = apply_controls(trials[actual], patch)
        return projected[key]
    result = dict(ids=ids, rows=rows, patches=dict(patches), scenes=replayed['selection'], totals=dict(totals),
                  trials={'baseline':trials['baseline'], **{name:projection(name,actual) for name,actual in aliases.items()}})
    if omitted_stage:
        entries = {name: sum(visits[aliases[name]].get(stage, 0) for stage in members(omitted_stage))
                   for name in ('enabled', 'omitted')}
        result['intervention'] = dict(stage=omitted_stage, enabled_visits=entries['enabled'],
                                      omitted_visits=entries['omitted'])
    if refinement:
        result.update(minimized=proposal(config,refinement.values), probe_count=refinement.count,
                      budget_reached=not refinement.finished)
        for field in ('controls','watch'):
            if field in patches['option']:
                result['minimized'][field] = patches['option'][field]
        result['trials']['minimized'] = apply_controls(trials[refinement.best],result['minimized'])
        result['patches']['minimized'] = result['minimized']
    return result


def describe(result, name):
    item = result['trials'][name]
    baseline = result['trials']['baseline']
    lines = ['Replay on retained scans: ' + ('both identities fully joined.' if item['joined'] else 'pair did not fully join.'),
             'Current YAML baseline: ' + ('pair fully joined.' if baseline['joined'] else 'pair did not fully join.'),
             item['attribution']]
    lines.append(strategy_summary(result, name))
    for key, state in item.get('watched', {}).items():
        lines.append(f"Witness {key}: {state}.")
    for person, separate in item.get('controls', {}).items():
        status = 'separate' if separate else 'unavailable' if separate is None else 'lost, split or absorbed'
        lines.append(f"Automatic control {person}: {status}.")
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

def strategy_summary(result, name):
    from .inspection_first_difference import changed, order
    before, after = result['trials']['baseline'], result['trials'][name]
    rows, ids = result['rows'], result['ids'][:2]
    first = min((k for k,r in rows.items() if r['source'] in ids and changed(k,before,after)),
                key=lambda k:(order(rows[k]),k), default=None)
    first_cue = 'No changed pair cue'
    if first is not None:
        r = rows[first]
        decision = after['decisions'].get(first,{})
        first_cue = f"Scene {r['scene']} cue {r['cue']} · {decision.get('route') or 'owner changed'}"
    joined = sum(after['partial'][p].get('joined target',0) for p in ids)
    total = sum(after['counts'][p] for p in ids)
    lost = sum(after['lost'][p] for p in ids)
    return f"{first_cue} · {joined}/{total} at opposite side's unique owner · {lost} pair unassigned · {len(after['outsiders'])} extra identities"
