"""Directional alternatives to forming a second core in the selected scene."""
from collections import Counter
import json

from .body_identity import available, enabled, policy as body_policy
from .body_observations import contract
from .clustering import matching_policy
from .inspection_advice import _number, _path
from .person_anchor import anchor_gate
from .pose_policy import anchor_sql
from .reference_roles import strict_sql, observation_sql
from .threshold_boundaries import at_or_above, strictly_above
from .visibility import cached as measured_visibility

FACE_GATES = {'pixels': 'clustering.min_anchor_face_pixels', 'sharpness': 'clustering.min_anchor_sharpness',
              'pose_deviation': 'clustering.max_anchor_pose_deviation',
              'visible_face_fraction': 'clustering.min_anchor_visible_face_fraction'}
BODY_GATES = {'pixels': 'sapiensid.identities.anchor_min_person_pixels',
              'sharpness': 'sapiensid.identities.anchor_min_sharpness',
              'keypoints': 'sapiensid.identities.anchor_min_keypoints'}


def _anchors(db, config, scan, ids, kind):
    grouping = config.data['clustering']
    if kind == 'face':
        sql = ('SELECT m.person,o.cue,o.id,o.quality FROM members m JOIN observations o ON o.id=m.observation '
               'WHERE m.person IN (?,?) AND o.scan=? AND o.rejected IS NULL AND o.embedding IS NOT NULL AND '
               + anchor_sql('o', grouping) + ' AND ' + strict_sql() +
               ' AND EXISTS (SELECT 1 FROM candidates c WHERE ' + observation_sql('c') +
               "=o.id AND c.scan=o.scan AND c.policy=? AND c.status='accepted' AND c.person=m.person)")
        for row in db.execute(sql, (*ids, scan, matching_policy(config))):
            quality = json.loads(row['quality'])
            if not measured_visibility(quality, config.data['detection']['faceocc_logit_threshold']):
                quality.pop('visible_face_fraction', None)
            yield row, quality, None
    elif enabled(config) and available(db):
        sql = ("SELECT m.person,o.cue,o.id,o.quality,m.proof FROM body_members m JOIN body_observations o "
               "ON o.id=m.observation WHERE m.person IN (?,?) AND o.scan=? AND o.contract=? AND m.policy=? "
               "AND m.status='accepted' AND m.role='anchor' AND o.rejected IS NULL AND o.embedding IS NOT NULL")
        for row in db.execute(sql, (*ids, scan, contract(config), body_policy(config))):
            gate = anchor_gate(row['quality'], config.data['sapiensid']['identities'], collection=config.data['sapiensid']['collection'])
            if gate['eligible']:
                proof = json.loads(row['proof'])
                support = proof.get('independent_support') if proof.get('reason') == 'body_scene_core' else None
                yield row, gate, support


def _profiles(db, config, scan, ids, kind, gates):
    profiles = {person: dict(count=0, measured=Counter(), peaks={}, support=0) for person in ids}
    for row, quality, support in _anchors(db, config, scan, ids, kind):
        profile = profiles[row['person']]
        profile['count'] += 1
        if type(support) is int:
            profile['support'] = max(profile['support'], support)
        for field in gates:
            value = quality.get(field)
            if _number(value):
                profile['measured'][field] += 1
                # Stable witness when several anchors share the same maximum.
                candidate = (value, row['cue'], row['id'])
                best = min if field == 'pose_deviation' else max
                profile['peaks'][field] = best(profile['peaks'].get(field, candidate), candidate)
    return profiles


def _face_birth_support(db, config, scan, ids):
    """Recount independent votes only for a still-identifiable local seed cohort."""
    from .independent_evidence import Votes
    votes = {person: Votes() for person in ids}
    seeded, mixed = set(), set()
    for row in db.execute(
        "SELECT c.person,o.scan,o.seconds,o.quality,json_extract(c.payload,'$.reason') AS reason,"
        "json_extract(c.payload,'$.face_comparison.stage') AS stage,json_extract(c.payload,'$.merged_from') AS merged "
        "FROM candidates c JOIN observations o ON o.id=" + observation_sql('c') +
        " AND o.scan=c.scan WHERE c.scan=? AND c.policy=? AND c.status='accepted' AND c.person IN (?,?)",
        (scan, matching_policy(config), *ids)):
        person = row['person']
        if row['merged'] or row['reason'] != 'candidate_group' or row['stage'] not in ('seed', 'match'):
            mixed.add(person)
        else:
            votes[person].add(row)
            if row['stage'] == 'seed':
                seeded.add(person)
    return {person: len(votes[person]) for person in seeded - mixed}


def alternatives(db, config, scan, ids, labels):
    routes = []
    by_source = {}
    for kind, gates in (('face', FACE_GATES), ('body', BODY_GATES)):
        profiles = _profiles(db, config, scan['id'], ids, kind, gates)
        if kind == 'face':
            for person, count in _face_birth_support(db, config, scan['id'], ids).items():
                profiles[person]['support'] = count
        for source, target in (ids, ids[::-1]):
            a, b = profiles[source], profiles[target]
            if not a['count'] or not b['count']:
                continue
            title = f"Prevent a separate {kind} core · {labels[source]} → {labels[target]}"
            notes = [f"Scene {scan['scene']}: {a['count']} eligible {kind} anchors for the source, "
                     f"{b['count']} for the target.",
                     "1. Exclude the source anchors from constructing this scene's core.",
                     "2. Keep an eligible target core. Other scenes and the other modality can still create an identity.",
                     ("3. Match the source as auxiliary against the surviving face bank. Similarity, margin, support and co-presence still apply."
                      if kind == 'face' else
                      "3. Match the source to a frozen body core or follow an assigned linked face. Check competitors and co-presence."),
                     "Rebuild both identities together. A withheld observation does not count as a successful attachment."]
            patches = []
            for field, key in gates.items():
                if a['measured'][field] != a['count'] or field not in b['peaks']:
                    continue
                peak, cue, witness = a['peaks'][field]
                surviving, other_cue, other_witness = b['peaks'][field]
                if field == 'pose_deviation':
                    cutoff = round(at_or_above(peak) - .000001, 6)
                    if surviving > cutoff:
                        continue
                else:
                    cutoff = strictly_above(peak, integer=field in ('pixels', 'keypoints'))
                    if surviving < cutoff:
                        continue
                changes = {key: cutoff}
                if kind == 'face':
                    changes['clustering.auxiliary_matching'] = True
                patch = _path(config, f'Demote source anchors by {field}', changes,
                    evidence=[f'Strongest source: {peak:g} at cue {cue} ({witness}).',
                              f'Surviving target: {surviving:g} at cue {other_cue} ({other_witness}).'])
                if patch['changes']:
                    patch['witnesses'] = [dict(label='Source and surviving target',refs=[(kind,witness),(kind,other_witness)])]
                    patches.append(patch)
            if 0 < a['support'] < min(8, b['support']):
                patch = _path(config, 'Require more evidence for a new scene core',
                    {('recognition.min_identity_support' if kind == 'face' else 'sapiensid.identities.min_support'): a['support']+1},
                    evidence=[f"Independent {kind} core support: source {a['support']}, target {b['support']}.",
                              "The source would lack support for independent confirmation. Reassignment still needs another route."])
                if patch['changes']:
                    patches.append(patch)
            if not patches:
                notes.append('No measured quality or support boundary separates these scene cores while preserving a target witness.')
            by_source.setdefault(source, {})[kind] = patches
            routes.append(dict(title=title, status='Rebuild hypothesis' if patches else 'No separating boundary',
                               notes=notes, patches=patches))
    for source, target in (ids, ids[::-1]):
        options = by_source.get(source, {})
        if options.get('face') and options.get('body'):
            combined = []
            for face in options['face']:
                for body in options['body']:
                    changes = {c['parameter']: c['proposed'] for p in (face, body) for c in p['changes']}
                    patch = _path(config, f"Face: {face['title']} + body: {body['title']}", changes,
                                  evidence=face['evidence'] + body['evidence'])
                    if patch['changes']:
                        patch['witnesses'] = face.get('witnesses',[]) + body.get('witnesses',[])
                        combined.append(patch)
            if combined:
                routes.append(dict(title=f"Prevent both separate cores · {labels[source]} → {labels[target]}",
                    status='Rebuild hypothesis', patches=combined,
                    notes=["The source has two ways to establish an independent identity in this scene.",
                           "Apply one measured face option and one body option together, then check attachment to the surviving target.",
                           "Replay counts withheld samples and checks final ownership."]))
    return routes
