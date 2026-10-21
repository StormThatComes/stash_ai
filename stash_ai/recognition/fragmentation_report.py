"""Read-only, evidence-based diagnostics for fragmented face identities."""
from __future__ import annotations

from contextlib import closing
from dataclasses import replace
import json
import math
from pathlib import Path
import sqlite3

from .clustering import Group, _similarities, matching_policy, vector, _groups, reference_admission, reference_context
from .config import plain
from .database import APPLICATION_ID, SCHEMA_VERSION, _reject_executable_schema
from .identity_merges import _merge_support_policy
from .models import namespace
from .pose_policy import anchor_sql
from .reference_roles import strict_sql
from .reference_usage import ReferenceUsage
from .reference_selection import select
from .scanning import scan_identity
from .reference_compatibility import fallback_sql
from .database import encoded


def _open_readonly(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'Face database does not exist: {path}')
    if path.stat().st_nlink != 1:
        raise ValueError('Recognition SQLite must not be a hard-linked file.')
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        if (db.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID
                or db.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION):
            raise ValueError('Unsupported facial recognition database/version.')
        _reject_executable_schema(db)
        db.execute('BEGIN')  # One consistent snapshot, including during another recognition run.
        return db
    except BaseException:
        db.close()
        raise


def _selected_scan(db, scene, model_namespace, policy, scope=None):
    scans = db.execute(
        "SELECT id,scope,scene,namespace,policy_sha FROM scans "
        "WHERE scene=? AND namespace=? AND status='complete' ORDER BY rowid DESC",
        (scene, model_namespace)).fetchall()
    if scope is not None:
        scans = [row for row in scans if row['scope'] == scope]
    if not scans:
        raise ValueError('No completed recognition scan for this scene, recognizer and scope.')
    scopes = {row['scope'] for row in scans}
    if len(scopes) != 1:
        raise ValueError('Scene ID is present in multiple Stash scopes; a scoped report is required.')
    scan = scans[0]
    if (db.execute('SELECT 1 FROM observations WHERE scan=? AND rejected IS NULL LIMIT 1',
                   (scan['id'],)).fetchone()
            and not db.execute('SELECT 1 FROM candidates WHERE scan=? AND policy=? LIMIT 1',
                               (scan['id'], policy)).fetchone()):
        raise ValueError('Latest scene scan has no grouping evidence for the current face configuration; '
                         'run recognition with this scene/configuration before reporting.')
    return scan


def _scene_identities(db, scan, policy):
    return [dict(row) for row in db.execute(
        'SELECT m.person AS identity_id,p.performer AS performer_id,COUNT(*) AS observations '
        'FROM candidates c JOIN observations o ON o.id=json_extract(c.payload,\'$.observation\') '
        'JOIN members m ON m.observation=o.id JOIN people p ON p.id=m.person '
        'WHERE c.scan=? AND c.policy=? AND c.status=\'accepted\' '
        'AND o.scan=c.scan AND o.rejected IS NULL '
        'GROUP BY m.person,p.performer ORDER BY m.person', (scan, policy))]


def _representatives(db, scan, person, maximum, grouping, *, bank=None, config=None):
    if bank is None:
        bank=_groups(db,scan['scope'],scan['namespace'],grouping,
            scan_policy=scan_identity(config,scan['namespace']) if config is not None else None,
            label_scenes=(scan['scene'],),compatibility=fallback_sql(db,config,scan['namespace']) if config is not None else None)
    group=next((group for group in bank if group.identifier==person),None)
    if group is None:
        return []
    rows={row['id']:row for row in db.execute('SELECT o.id,o.cue,o.seconds,o.quality,s.scene '
        'FROM observations o JOIN scans s ON s.id=o.scan WHERE o.id IN '
        '(SELECT value FROM json_each(?))',(encoded(group.sample_ids),))}
    return [(dict(id=identifier,scene=rows[identifier]['scene'],cue=rows[identifier]['cue'],
        seconds=rows[identifier]['seconds'],quality=_quality(rows[identifier]['quality']),
        reference_role='strict'),sample) for identifier,sample in zip(group.sample_ids,group.samples)]


QUALITY_METRICS = ('pixels', 'pose_deviation', 'sharpness', 'visible_face_fraction')


def _quality(raw):
    """Expose only finite measured quality signals; never dump crops or embeddings."""
    try:
        values = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        values = {}
    if not isinstance(values, dict):
        values = {}
    return {name: float(value) if type(value) in (float, int) and math.isfinite(value) else None
            for name in QUALITY_METRICS for value in (values.get(name),)}


def _quality_summary(qualities):
    """Missing/legacy quality fields remain visibly missing, not zero."""
    result = {}
    for metric in QUALITY_METRICS:
        values = sorted(item[metric] for item in qualities if item[metric] is not None)
        result[metric] = {
            'count': len(values), 'missing': len(qualities) - len(values),
            'minimum': round(values[0], 6) if values else None,
            'median': round((values[(len(values) - 1) // 2] + values[len(values) // 2]) / 2, 6)
            if values else None,
            'p90': round(values[math.ceil(.9 * len(values)) - 1], 6) if values else None,
            'maximum': round(values[-1], 6) if values else None,
        }
    return result


def _scene_quality(db, scan):
    buckets = {}
    for row in db.execute('SELECT quality,rejected FROM observations WHERE scan=? ORDER BY cue,id',
                          (scan,)):
        label = row['rejected'] or 'retained'
        buckets.setdefault(label, []).append(_quality(row['quality']))
    return {label: {'observations': len(qualities), 'metrics': _quality_summary(qualities)}
            for label, qualities in sorted(buckets.items())}


def _rejected_examples(db, scan, per_reason=8):
    """Bounded audit pointers into the VTT without exporting original images."""
    examples = {}
    rows = db.execute(
        'SELECT id,cue,seconds,rejected,quality FROM observations '
        'WHERE scan=? AND rejected IS NOT NULL ORDER BY cue,id', (scan,))
    for row in rows:
        selected = examples.setdefault(row['rejected'], [])
        if len(selected) < per_reason:
            selected.append({'observation_id': row['id'], 'cue': row['cue'],
                             'seconds': row['seconds'], 'quality': _quality(row['quality'])})
    return examples


def _reference_details(db, scan, person, representatives, limit, grouping, config):
    anchor=reference_admission(grouping,scan_policy=scan_identity(config,scan['namespace']),label_scenes=(scan['scene'],),compatibility=fallback_sql(db,config,scan['namespace']) if config is not None else None)+' AND '+strict_sql()
    total = db.execute(
        'SELECT COUNT(*) FROM observations o JOIN members m ON m.observation=o.id '
        'JOIN scans s ON s.id=o.scan WHERE m.person=? AND s.scope=? AND s.namespace=? '
        f"AND s.status='complete' AND o.rejected IS NULL AND {anchor}",
        (person, scan['scope'], scan['namespace'])).fetchone()[0]
    scenes = {}
    for info, _ in representatives:
        scenes[info['scene']] = scenes.get(info['scene'], 0) + 1
    return {
        'available': total, 'selected': len(representatives), 'limit': limit,
        'truncated': total > len(representatives), 'selection': 'production_quality_diversity_v1',
        'selected_by_scene': scenes,
        'quality': _quality_summary([info['quality'] for info, _ in representatives]),
        'samples': [info for info, _ in representatives],
    }


def _strict_scene_counts(db, scan, policy, grouping):
    anchor = anchor_sql('o', grouping)
    """Counts are only possible anchors; the local pass revalidates strict cores."""
    result = {}
    rows = db.execute(
        "SELECT c.person,COUNT(DISTINCT o.seconds) AS strict_cues,"
        "COUNT(*) AS strict_observations FROM candidates c "
        "JOIN observations o ON o.id=json_extract(c.payload,'$.observation') "
        "WHERE c.scan=? AND c.policy=? AND c.status='accepted' "
        "AND o.scan=c.scan AND o.rejected IS NULL "
        "AND json_extract(c.payload,'$.reason')<>'same_scene_identity' "
        f"AND {anchor} AND EXISTS (SELECT 1 FROM members m WHERE m.observation=o.id AND m.person=c.person AND {strict_sql()}) GROUP BY c.person", (scan, policy))
    for row in rows:
        result[row['person']] = dict(strict_cues=row['strict_cues'],
                                      strict_observations=row['strict_observations'])
    return result


def _score(left, right, scene=None):
    """Equivalent to the flat-group cross mean/minimum checks; not a replay."""
    import numpy as np
    first = np.stack([sample for _, sample in left])
    second = np.stack([sample for _, sample in right])
    if first.shape[1] != second.shape[1]:
        raise ValueError('Incompatible recognition embedding dimensions.')
    values = np.asarray([[float(np.dot(a,b)) for b in second] for a in first])
    low = np.unravel_index(int(values.argmin()), values.shape)
    # Sum converted float32 dot scores in float64, as the matching code does.
    mean = math.fsum(float(value) for value in values.flat) / values.size
    if scene is None:
        return mean, float(values[low]), (left[low[0]][0], right[low[1]][0]), None
    details = {
        'cross_scores': {
            'comparisons': int(values.size),
            'p10': round(float(np.quantile(values, .1)), 6),
            'median': round(float(np.median(values)), 6),
            'p90': round(float(np.quantile(values, .9)), 6),
        },
        'worst_pairs': [
            {'similarity': round(float(values.flat[index]), 6),
             'left': left[np.unravel_index(int(index), values.shape)[0]][0],
             'right': right[np.unravel_index(int(index), values.shape)[1]][0]}
            for index in np.argsort(values, axis=None)[:3]
        ],
    }
    if scene is not None:
        first_scene = [entry for entry in left if entry[0]['scene'] == scene]
        second_scene = [entry for entry in right if entry[0]['scene'] == scene]
        if first_scene and second_scene:
            scene_values = np.stack([sample for _, sample in first_scene]) @ np.stack(
                [sample for _, sample in second_scene]).T
            details['selected_scene_only'] = {
                'left_representatives': len(first_scene),
                'right_representatives': len(second_scene),
                'cross_mean': round(float(np.mean(scene_values, dtype=np.float64)), 6),
                'cross_minimum': round(float(scene_values.min()), 6),
                'comparisons': int(scene_values.size),
            }
        else:
            details['selected_scene_only'] = {
                'left_representatives': len(first_scene),
                'right_representatives': len(second_scene),
                'cross_mean': None, 'cross_minimum': None, 'comparisons': 0,
            }
    return mean, float(values[low]), (left[low[0]][0], right[low[1]][0]), details


def _co_present(db, left, right):
    return db.execute(
        'SELECT 1 FROM members a JOIN observations x ON x.id=a.observation '
        'JOIN observations y ON y.scan=x.scan AND y.cue=x.cue '
        'JOIN members b ON b.observation=y.id '
        'WHERE a.person=? AND b.person=? AND x.id<>y.id '
        'AND x.rejected IS NULL AND y.rejected IS NULL LIMIT 1',
        (left, right)).fetchone() is not None


def _recorded_support(db, scan, left, right, references, config, *, local=False):
    """Recheck stored nominations; absence is not evidence of failed proposals."""
    if local:
        local_data = plain(config.data)
        bonus = local_data['clustering']['same_scene_match_bonus']
        local_data['recognition']['match_threshold'] = max(0., local_data['recognition']['match_threshold'] - bonus)
        local_data['clustering']['cohesion_threshold'] = max(0., local_data['clustering']['cohesion_threshold'] - bonus)
        config = replace(config, data=local_data)
    policy = _merge_support_policy(config, scene_scan=scan['id'] if local else None)
    settings, clustering = config.data['recognition'], config.data['clustering']
    rows = db.execute(
        'SELECT o.id,o.seconds,o.dimensions,o.embedding,s.scene,s.id AS selected_scan_id FROM merge_support v '
        'JOIN observations o ON o.id=v.observation JOIN scans s ON s.id=o.scan '
        'WHERE v.policy=? AND ((v.source=? AND v.target=?) OR (v.source=? AND v.target=?)) '
        'AND s.scope=? AND s.namespace=? AND s.policy_sha=? AND s.status=\'complete\' '
        'AND o.rejected IS NULL AND NOT EXISTS ('
        ' SELECT 1 FROM scans newer WHERE newer.scope=s.scope AND newer.namespace=s.namespace '
        ' AND newer.scene=s.scene AND newer.status=\'complete\' AND newer.rowid>s.rowid) '
        'ORDER BY CAST(s.scene AS INTEGER),s.scene,o.seconds,o.id',
        (policy, left, right, right, left, scan['scope'], scan['namespace'], scan['policy_sha']))
    groups = [Group(person, [sample for _, sample in references[person]], True)
              for person in (left, right)]
    cues = set()
    nominations = 0
    for row in rows:
        if local and row['selected_scan_id'] != scan['id']:
            continue
        nominations += 1
        cue = (row['scene'], row['seconds'])
        if cue in cues:
            continue
        face = vector(row)
        if all((mean >= settings['match_threshold'] and low >= clustering['cohesion_threshold'])
               for mean, low in (_similarities(group, face) for group in groups)):
            cues.add(cue)
    return {'stored_nominations': nominations, 'validated_nominations': len(cues),
            'required': settings['min_merge_support'], 'complete_replay_proof': False,
            'note': 'Only stored nominations surviving a bounded representative check; '
                    'missing nominations are not proof that a merge would fail.'}


def _local_pair(pair, first, second, config, strict_counts):
    settings, clustering = config.data['recognition'], config.data['clustering']
    bonus = clustering['same_scene_match_bonus']
    match = max(0., settings['match_threshold'] - bonus)
    cohesion = max(0., clustering['cohesion_threshold'] - bonus)
    minimum = clustering['same_scene_min_support']
    left, right = strict_counts.get(first, {}), strict_counts.get(second, {})
    left_cues, right_cues = left.get('strict_cues', 0), right.get('strict_cues', 0)
    blockers = []
    if bonus <= 0:
        blockers.append('same_scene_context_disabled')
    if left_cues == 0 or right_cues == 0:
        blockers.append('both_identities_need_strict_accepted_scene_observations')
    if max(left_cues, right_cues) < minimum:
        blockers.append('no_possible_strict_local_core_by_cue_count')
    if pair['cross_mean'] < match:
        blockers.append('local_cross_mean_below_match_threshold')
    if pair['cross_minimum'] < cohesion:
        blockers.append('local_cross_sample_below_cohesion_threshold')
    if pair['co_present_observed'] and pair['cross_minimum'] < settings['cooccurrence_merge_threshold']:
        blockers.append('local_co_present_requires_stronger_cross_similarity')
    return {'effective_match_threshold': round(match, 6),
            'effective_cohesion_threshold': round(cohesion, 6),
            'required_strict_core_cues': minimum,
            'strict_accepted_scene_cues': {'left': left_cues, 'right': right_cues},
            'possible_core_by_count': max(left_cues, right_cues) >= minimum,
            'numeric_and_count_blockers': blockers,
            'strict_core_revalidation_complete': False,
            'can_merge': None,
            'note': 'Only necessary numeric and raw cue-count tests. Strict cores are '
                    'revalidated against all identities in a cold replay; nominations, '
                    'constituents, co-presence and replay ordering may still prevent merging.'}


def build(config, scene_id, *, limit=20, min_mean=None, scope=None, compare_scene_ids=None):
    """Read only committed state; returned suggestions do not promise a cold-replay merge."""
    if not 1 <= limit <= 500:
        raise ValueError('Report limit must be 1..500.')
    settings, clustering = config.data['recognition'], config.data['clustering']
    if min_mean is None:
        min_mean = settings['match_threshold'] - .20
    if not -1 <= min_mean <= 1 or not math.isfinite(min_mean):
        raise ValueError('Report min-mean must be between -1 and 1.')
    policy = matching_policy(config)
    with closing(_open_readonly(config.database)) as db:
        scan = _selected_scan(db, str(scene_id), namespace(config), policy, scope=scope)
        identities = _scene_identities(db, scan['id'], policy)
        bank=_groups(db,scan['scope'],scan['namespace'],clustering,
            scan_policy=scan_identity(config,scan['namespace']),label_scenes=(scan['scene'],),compatibility=fallback_sql(db,config,scan['namespace']) if config is not None else None)
        references = {person['identity_id']: _representatives(
            db, scan, person['identity_id'], clustering['max_samples_per_person'], clustering,bank=bank) for person in identities}
        strict_counts = _strict_scene_counts(db, scan['id'], policy, clustering)
        for person in identities:
            identifier = person['identity_id']
            cues = strict_counts.get(identifier, {}).get('strict_cues', 0)
            person['strict_accepted_scene_cues'] = cues
            person['reference_status']='available' if references[identifier] else 'no_strict_references'
            person['same_scene_anchor_possible_by_count'] = (
                cues >= clustering['same_scene_min_support'])
            anchor = anchor_sql('o', clustering)
            anchored, auxiliary = db.execute(
                f'SELECT COUNT(*) FILTER (WHERE {anchor}),COUNT(*) FILTER (WHERE NOT ({anchor})) '
                'FROM members m JOIN observations o ON o.id=m.observation '
                'WHERE o.scan=? AND m.person=? AND o.rejected IS NULL',
                (scan['id'], identifier)).fetchone()
            person['pose_roles'] = {'anchor': anchored, 'auxiliary': auxiliary}
            person['representatives'] = _reference_details(
                db, scan, identifier, references[identifier], clustering['max_samples_per_person'], clustering,config)
        decisions = [dict(row) for row in db.execute(
            "SELECT json_extract(payload,'$.reason') AS reason,status,COUNT(*) AS count "
            "FROM candidates WHERE scan=? AND policy=? "
            "GROUP BY json_extract(payload,'$.reason'),status ORDER BY count DESC,reason",
            (scan['id'], policy))]
        total, rejected = db.execute(
            'SELECT COUNT(*),COUNT(*) FILTER (WHERE rejected IS NOT NULL) '
            'FROM observations WHERE scan=?', (scan['id'],)).fetchone()
        rejected_reasons = [dict(row) for row in db.execute(
            'SELECT rejected AS reason,COUNT(*) AS count FROM observations '
            'WHERE scan=? AND rejected IS NOT NULL GROUP BY rejected ORDER BY count DESC,rejected',
            (scan['id'],))]
        withheld_examples = [dict(row) for row in db.execute(
            "SELECT o.id AS observation_id,o.cue,o.seconds,"
            "json_extract(c.payload,'$.reason') AS reason,"
            "json_extract(c.payload,'$.scores') AS score_json, "
            "json_extract(c.payload,'$.global_identity_decision') AS global_identity_decision, "
            "json_extract(c.payload,'$.decision_conflict') AS decision_conflict, "
            "json_extract(c.payload,'$.cohort_resolution') AS cohort_resolution, "
            "json_extract(c.payload,'$.provisional_pool') AS provisional_pool, "
            "json_extract(c.payload,'$.global_view_consolidation') AS global_view_consolidation "
            "FROM candidates c JOIN observations o ON o.id=json_extract(c.payload,'$.observation') "
            "WHERE c.scan=? AND c.policy=? AND c.status='withheld' "
            "AND o.scan=c.scan ORDER BY o.cue,o.seconds,o.id LIMIT 50",
            (scan['id'], policy))]
        for example in withheld_examples:
            example['scores'] = json.loads(example.pop('score_json') or '[]')
            for field in ('global_identity_decision','decision_conflict','cohort_resolution','provisional_pool','global_view_consolidation'):
                example[field]=json.loads(example[field]) if example[field] else None
        people = {item['identity_id']: item for item in identities}
        ids = list(people)
        pairs = []
        for n, left in enumerate(ids):
            for right in ids[n + 1:]:
                if not references[left] or not references[right]:
                    continue
                mean, minimum, worst, _ = _score(references[left], references[right])
                if mean < min_mean:
                    continue
                blockers = []
                if mean < settings['match_threshold']:
                    blockers.append('cross_mean_below_match_threshold')
                if minimum < clustering['cohesion_threshold']:
                    blockers.append('cross_sample_below_cohesion_threshold')
                pairs.append({
                    'left': people[left], 'right': people[right],
                    'cross_mean': round(mean, 6), 'cross_minimum': round(minimum, 6),
                    'worst_representatives': {'left': worst[0], 'right': worst[1]},
                    'blockers': blockers,
                    'indicative_thresholds': {
                        'match_threshold_at_most': round(mean, 6),
                        'cohesion_threshold_at_most': round(minimum, 6)},
                })
        pairs.sort(key=lambda pair: (-pair['cross_mean'], -pair['cross_minimum'],
                                     pair['left']['identity_id'], pair['right']['identity_id']))
        views=()
        if clustering['anchor_view_matching']:
            from .global_views import catalogue
            usage=ReferenceUsage(clustering['max_reference_bytes'])
            for group in bank:
                usage.add(group.samples,people=1,kind='Report strict')
            views=catalogue(db,config,scan['scope'],scan['namespace'],usage=usage,
                scan_policy=scan_identity(config,scan['namespace']),complete=True)
        from .global_views import consolidation_plan
        consolidation_decisions=consolidation_plan(db,views,config)
        from .historical_references import History
        from .reference_registry import resolve as registry_plan
        registry_usage=ReferenceUsage(clustering['max_reference_bytes'])
        for group in bank:
            registry_usage.add(group.samples,kind='Report registry strict')
        for view in views:
            registry_usage.add(view.examples,kind='Report registry views')
        labels={r['id']:tuple(r) for r in db.execute('SELECT * FROM people WHERE id IN '
            '(SELECT value FROM json_each(?))',(encoded([g.identifier for g in bank]),))}
        registry=History(tuple(bank),views,labels,reference_context(bank),views)
        registry_plan(db,registry,config,registry_usage)
        component_by_person={person:decision for decision in consolidation_decisions for person in decision['people']}
        selected_pairs = pairs[:limit]
        view_owners={}
        for view in views:
            view_owners.setdefault(view.group.identifier,[]).append(view)
        for pair in selected_pairs:
            a, b = pair['left']['identity_id'], pair['right']['identity_id']
            pair['score_diagnostics'] = _score(references[a], references[b], scan['scene'])[3]
            co_present = _co_present(db, a, b)
            if co_present and pair['cross_minimum'] < settings['cooccurrence_merge_threshold']:
                pair['blockers'].append('co_present_requires_stronger_cross_similarity')
            pair['co_present_observed'] = co_present
            pair['recorded_merge_support'] = _recorded_support(db, scan, a, b, references, config)
            pair['same_scene_merge'] = _local_pair(pair, a, b, config, strict_counts)
            pair['same_scene_merge']['recorded_merge_support'] = _recorded_support(
                db, scan, a, b, references, config, local=True)
            from .global_views import cohort_evidence
            evidence=[cohort_evidence(db,left,right,config)
                for left in view_owners.get(a,()) for right in view_owners.get(b,())]
            pair['confirmed_view_evidence']=(max(evidence,key=lambda p:(not p['blockers'],p['mean_cross_similarity']))
                if evidence else None)
            left_plan,right_plan=component_by_person.get(a),component_by_person.get(b)
            same=left_plan is not None and left_plan is right_plan
            pair['global_view_component']=(
                {k:v for k,v in left_plan.items() if k!='people'} if same else
                dict(outcome='blocked',blocker='different_strong_components'))
            pair['can_merge'] = None  # Full frozen-component proof still requires replay.
        anchor = anchor_sql('o', clustering)
        anchored, auxiliary = db.execute(
            f'SELECT COUNT(*) FILTER (WHERE rejected IS NULL AND {anchor}),' 
            f'COUNT(*) FILTER (WHERE rejected IS NULL AND NOT ({anchor})) '
            'FROM observations o WHERE scan=?', (scan['id'],)).fetchone()
        from .cross_scene_report import compare
        comparisons = (compare(db, scan, compare_scene_ids, policy, config, limit=limit)
                       if compare_scene_ids else [])
        multiview = _view_attempt_summary(db, scan['id'], policy, scan['scope'], scan['namespace'])
        return {
            'reference_registry':dict(format=1,frozen=True,aliases=registry.aliases,proposed_updates=registry.actions),
            'reference_context':reference_context(bank),
            'reference_selection':'production_quality_diversity_v1',
            'cross_scene_comparisons':comparisons,
            'multi_view_diagnostics':multiview,
            'format': 1, 'kind': 'recognition_fragmentation',
            'scene_id': str(scene_id), 'scope': scan['scope'], 'scan_id': scan['id'],
            'namespace': scan['namespace'], 'matching_policy': policy,
            'thresholds': {
                'match_threshold': settings['match_threshold'],
                'cohesion_threshold': clustering['cohesion_threshold'],
                'ambiguity_margin': settings['ambiguity_margin'],
                'cooccurrence_merge_threshold': settings['cooccurrence_merge_threshold'],
                'min_merge_support': settings['min_merge_support'],
                'same_scene_match_bonus': clustering['same_scene_match_bonus'],
                'same_scene_min_support': clustering['same_scene_min_support'],
                'max_samples_per_person': clustering['max_samples_per_person'],
                'max_anchor_pose_deviation': clustering['max_anchor_pose_deviation'],
                'auxiliary_matching': clustering['auxiliary_matching'],
                'auxiliary_min_support': clustering['auxiliary_min_support'],
                'local_match_threshold': round(max(0., settings['match_threshold'] -
                                                  clustering['same_scene_match_bonus']), 6),
                'local_cohesion_threshold': round(max(0., clustering['cohesion_threshold'] -
                                                     clustering['same_scene_match_bonus']), 6)},
            'summary': {'observations': total, 'rejected': rejected,
                         'pose_roles': {'anchor': anchored, 'auxiliary': auxiliary},
                        'identities': len(identities), 'pairs_checked': len(ids)*(len(ids)-1)//2,
                        'pairs_above_min_mean': len(pairs), 'pairs_reported': len(selected_pairs)},
            'decisions': decisions, 'rejected_reasons': rejected_reasons,
            'rejected_examples': _rejected_examples(db, scan['id']),
            'observation_quality': _scene_quality(db, scan['id']),
            'detection_gates': {key: config.data['detection'][key] for key in (
                'score_threshold', 'min_face_pixels', 'min_sharpness', 'max_pose_deviation',
                'min_visible_face_fraction', 'require_complete_face')},
            'withheld_examples': withheld_examples, 'identities': identities,
            'potential_fragments': selected_pairs,
            'limitations': [
                'Pair checks use current bounded references, not a complete cold replay.',
                'A successful pairwise comparison does not guarantee an identity merge; '
                'nominations, co-presence, ambiguity and constituent history still matter.',
                'Only identities assigned to the selected scene are compared; withheld '
                'observations appear in the decisions summary, not as separate identities.',
                'Recorded merge support is diagnostic, not a proof of future merging.',
                'Local core counts are unvalidated upper bounds; competing identity checks '
                'and chronological merge evidence require a full replay.',
                'Representatives use production strict authority, compatibility and quality/diversity selection. '
                'Attachments remain visible as memberships, but do not train strict reference scores.',
                'Missing quality metrics mean legacy/unmeasured data, not zero quality.',
            ],
        }


def _view_attempt_summary(db, scan, policy, scope, namespace):
    """Persisted per-observation vetoes, not inferred from successful identities."""
    result={}
    for stage in ('anchor_view','auxiliary_view'):
        rows=db.execute(
            "SELECT COALESCE(json_extract(payload,?), 'not_evaluated') reason,COUNT(*) "
            'FROM candidates WHERE scan=? AND policy=? GROUP BY reason ORDER BY reason',
            (f'$.{stage}_probe',scan,policy))
        result[stage]={reason:count for reason,count in rows}
        column='anchor_view_seeded' if stage=='anchor_view' else 'auxiliary_view_seeded'
        seeded=db.execute(
            "SELECT COALESCE(SUM(json_extract(payload,?)),0) FROM candidates "
            'WHERE scan=? AND policy=?',
            (f'$.{column}',scan,policy)).fetchone()[0]
        result[stage]['prototypes_seeded_in_scene']=int(seeded)
        if stage=='auxiliary_view':
            # Confirmed auxiliary views are global within the selected replay;
            # prototypes created in other scenes may also serve this scene.
            global_count=db.execute(
                "SELECT COALESCE(SUM(json_extract(c.payload,?)),0) FROM candidates c "
                'JOIN scans s ON s.id=c.scan WHERE c.policy=? AND s.scope=? AND s.namespace=?',
                (f'$.{column}',policy,scope,namespace)).fetchone()[0]
            result[stage]['prototypes_seeded_in_scope']=int(global_count)
    return result


def execute(args, config):
    result = build(config, args.scene_id, limit=args.limit, min_mean=args.min_mean, scope=args.scope,
                   compare_scene_ids=getattr(args,'compare_scene_id',None))
    output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output is None:
        print(output, end='')
    else:
        target = Path(args.output).expanduser().resolve()
        if target == config.database.resolve():
            raise ValueError('Report output must not overwrite the face database.')
        with target.open('x', encoding='utf-8') as file:
            file.write(output)
        print(f'Face fragmentation report written: {target}')
    return 0
