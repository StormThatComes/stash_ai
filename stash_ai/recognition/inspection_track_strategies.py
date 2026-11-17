"""Cue-ordered pair hypotheses from the same geometry and embeddings as body tracking."""

from collections import defaultdict
import json

import numpy as np

from .body_identity import Scores, _load_scene, available, enabled, policy, quality
from .body_tracklets import paths
from .inspection_advice import _path
from .inspection_pair_evidence import witness
from .inspection_runtime import check_cancelled
from .merge_report_policy import parameter
from .threshold_boundaries import at_or_below


LOCAL = 'sapiensid.identities.'
TRACK = LOCAL + 'track_similarity_threshold'
TITLES = {
    'turn': 'Follow the body through a turn between compatible faces',
    'crossing': 'Compare both assignments through two-person contact',
    'gap': 'Recover an unassigned observation between A and B',
}


def _route(key):
    return dict(
        strategy=key, title=TITLES[key], status='Insufficient evidence', witnesses=[], patches=[], notes=[]
    )


def _track_option(config, title, low):
    gates = {
        LOCAL + 'extended_track_graph': True,
        LOCAL + 'tracklet_matching': True,
        LOCAL + 'body_first_matching': True,
        LOCAL + 'endpoint_matching': True,
        LOCAL + 'turn_recovery': True,
    }
    if low is not None and 0 <= low < parameter(config.data, TRACK):
        gates[TRACK] = at_or_below(low)
    return _path(config, title, gates)


def _fragments(chain, strengths, owners, ids, *, faces):
    left = None
    for i, row in enumerate(chain):
        owner = row['face_person'] if faces and row.get('face_anchor') else None
        if not faces:
            owner = owners.get(row['id'])
        if owner is None:
            continue
        if left is not None and {left[1], owner} == set(ids):
            start = left[0]
            yield chain[start : i + 1], strengths[start:i]
        left = i, owner


def _describe_chain(config, route, segment, edges, owners, face_score, frames, scores):
    a, b = segment[0], segment[-1]
    route['status'] = 'Rebuild hypothesis'
    minimum = min(edges)
    cutoff = parameter(config.data, TRACK)
    first = next((i for i, score in enumerate(edges) if score < cutoff), None)
    route['notes'].append(
        f'Body path across cues {a["cue"]}–{b["cue"]}, minimum {minimum:.4f} / {cutoff:.4f}.'
    )
    route['witnesses'].append(witness('Track endpoints', 'body', a['id'], b['id']))
    if first is not None:
        left, right = segment[first : first + 2]
        route['notes'].append(f'First similarity blocker at cue {right["cue"]}: {edges[first]:.4f}.')
        route['witnesses'].append(witness('First blocked edge', 'body', left['id'], right['id']))
        peers = [r for r in frames[right['cue']] if r['id'] != right['id']]
        if peers:
            rival = max(peers, key=lambda r: scores.face_similarity(left['_vector'], r['_vector']))
            similarity = scores.face_similarity(left['_vector'], rival['_vector'])
            route['notes'].append(
                f'Closest other body in cue {right["cue"]}: {similarity:.4f} before geometry gates.'
            )
            route['witnesses'].append(witness('Competing continuation', 'body', left['id'], rival['id']))
    else:
        route['notes'].append('The retained path already clears the track similarity threshold.')
    if face_score is not None:
        route['notes'].append(f'Endpoint face similarity {face_score:.4f}.')
        route['witnesses'].append(witness('Face endpoints', 'face', a['face'], b['face']))
    missing = [r for r in segment[1:-1] if owners.get(r['id']) is None]
    option = _track_option(config, 'Replay the measured body path', minimum)
    option['watch'] = [r['id'] for r in missing]
    for row in missing:
        route['witnesses'].append(
            dict(label=f'Unassigned body at cue {row["cue"]}', refs=[('body', row['id'])])
        )
    route['patches'].append(option)


def _contact_paths(frames, before, after, scores, cutoff, margin):
    """Challenge each endpoint mapping with the production two-track assignment."""
    from .crossing_tracks import crossings
    from .perspective_shots import change
    from .track_geometry import rejection, torso_points, translation

    roots = {r['id']: owner for endpoint in (before, after) for owner, r in enumerate(endpoint, 1)}
    decisions, geometry = {}, {}
    hidden = []
    for cue, frame in frames.items():
        if len(frame) == 1:
            hidden.append(cue)
        for row in frame:
            q = quality(row)
            person = json.loads(row['geometry'])
            signature = q.get('frame_signature')
            if not person.get('bounds') or not signature:
                return {}
            pixels = np.frombuffer(bytes.fromhex(signature), np.uint8).astype(float)
            if pixels.size != 48:
                return {}
            geometry[row['id']] = person['bounds'], torso_points(person), q, pixels.reshape(-1, 3)
            decisions[row['id']] = dict(person=roots.get(row['id']))

    def score(left, right, raw=False):
        a, ta, qa, pa = geometry[left['id']]
        b, tb, qb, pb = geometry[right['id']]
        if qa['view_shot'] != qb['view_shot'] or change(pa, pb, True) != 'continuous':
            return None
        if rejection(a, b, ta, tb, translation=translation(qa, qb, pa, pb)):
            return None
        value = scores.face_similarity(left['_vector'], right['_vector'])
        return value if raw or value >= cutoff else None

    return crossings(
        frames,
        roots,
        decisions,
        score,
        margin,
        occluded=hidden,
        raw=lambda a, b: score(a, b, True),
        visible_single=lambda r: len(geometry[r['id']][1]) >= 3 and not quality(r).get('query_only'),
    )


def _crossing(config, ids, rows, scores):
    route = _route('crossing')
    by_cue = defaultdict(list)
    for row in rows:
        by_cue[row['cue']].append(row)
    # Two strict face endpoints on either side constrain the two physical assignments.
    for cue, before in sorted(by_cue.items()):
        check_cancelled()
        if len(before) != 2 or not all(r['face_anchor'] and r['_face_vector'] is not None for r in before):
            continue
        start = {r['face_person'] for r in before}
        if len(start) != 2 or not start.intersection(ids):
            continue
        shot = quality(before[0]).get('view_shot')
        if shot is None or quality(before[1]).get('view_shot') != shot:
            continue
        interior = []
        contact = False
        # Same bounded contact window as the production two-person crossing resolver.
        for end in range(cue + 1, cue + 13):
            frame = by_cue.get(end, [])
            if len(frame) not in (1, 2) or any(quality(r).get('view_shot') != shot for r in frame):
                break
            if len(frame) == 2 and all(r['face_anchor'] and r['_face_vector'] is not None for r in frame):
                finish = {r['face_person'] for r in frame}
                common = start & finish
                copresent = start == finish == set(ids)
                if not contact or not (copresent or len(common) == 1 and (start ^ finish) == set(ids)):
                    break
                partner = ids[1] if copresent else next(iter(common))
                before = sorted(before, key=lambda r: r['face_person'] == partner)
                after = sorted(frame, key=lambda r: r['face_person'] == partner)
                rankings = []
                assignments = []
                frames = {index: by_cue[index] for index in range(cue, end + 1)}
                cutoff = parameter(config.data, TRACK)
                margin = max(0.025, config.data['sapiensid']['identities']['ambiguity_margin'])
                for title, order in [('Keep the partner', after), ('Swap with partner', after[::-1])]:
                    face = [
                        scores.face_similarity(a['_face_vector'], b['_face_vector'])
                        for a, b in zip(before, order)
                    ]
                    body = [
                        scores.face_similarity(a['_vector'], b['_vector']) for a, b in zip(before, order)
                    ]
                    claims = _contact_paths(frames, before, order, scores, cutoff, margin)
                    assignments.append(claims)
                    rankings.append(min(face))
                    route['notes'].append(
                        f'{title}: weakest endpoint face {min(face):.4f}, body {min(body):.4f}; '
                        f'{len(claims)} interior poses resolved by the crossing planner.'
                    )
                    for i, (a, b) in enumerate(zip(before, order), 1):
                        route['witnesses'].append(witness(f'{title} · track {i}', 'body', a['id'], b['id']))
                route['notes'].insert(0, f'Contact in cues {cue}–{end}, co-present partner {partner}.')
                cfg = config.data['recognition']
                if copresent:
                    route['status'] = 'Co-present identities'
                    route['notes'].append(
                        'Both selected identities have distinct simultaneous body endpoints. '
                        'Inspect their interior assignments before considering a merge.'
                    )
                elif (
                    rankings[0] >= cfg['match_threshold']
                    and rankings[0] - rankings[1] >= cfg['ambiguity_margin']
                    and rankings[0] > rankings[1]
                ):
                    route['status'] = 'Rebuild hypothesis'
                    patch = _track_option(
                        config, 'Replay crossing resolution with both face endpoints', None
                    )
                    patch.update(controls=[partner], watch=[r['id'] for r in interior])
                    route['patches'].append(patch)
                else:
                    route['status'] = 'Ambiguous endpoints'
                    route['notes'].append('The preserved partner lacks a unique face-supported assignment.')
                for row in interior:
                    mappings = [
                        claims[row['id']][0] if row['id'] in claims else None for claims in assignments
                    ]
                    if any(mappings):
                        labels = [
                            'pair' if owner == 1 else 'partner' if owner == 2 else 'unresolved'
                            for owner in mappings
                        ]
                        route['notes'].append(
                            f'Cue {row["cue"]} body {row["id"]}: keep / swap = ' + ' / '.join(labels)
                        )
                return route
            contact |= len(frame) == 1 or any(
                (quality(r).get('overlap') or 0) > 0
                or quality(r).get('query_only_reason') == 'unresolved_partner'
                for r in frame
            )
            interior.extend(frame)
    route['notes'].append(
        'No same-shot contact bracketed by two distinct measured face owners on each side.'
    )
    return route


def track_strategies(db, config, scan, ids):
    turn, gap = _route('turn'), _route('gap')
    if not enabled(config) or not available(db):
        return [turn, _route('crossing'), gap]
    try:
        record, rows, _, _, _ = _load_scene(db, scan['id'], config)
    except ValueError as exc:
        routes = [turn, _route('crossing'), gap]
        for route in routes:
            route['notes'].append(str(exc))
        return routes
    owners = dict(
        db.execute(
            'SELECT m.observation,m.person FROM body_members m JOIN body_observations b '
            "ON b.id=m.observation WHERE b.scan=? AND m.policy=? AND m.status='accepted'",
            (scan['id'], policy(config)),
        )
    )
    settings = config.data['sapiensid']['identities']
    scores = Scores(config)
    crossing = _crossing(config, ids, rows, scores)
    best = {}
    for threshold in dict.fromkeys((parameter(config.data, TRACK), 0.0)):
        candidates = paths(
            rows,
            dict(settings, track_similarity_threshold=threshold),
            scores,
            record['occupied_cues'],
            extended=True,
            gap_boxes=record['gap_boxes'],
            motion=record['motion'],
        )
        for chain, strengths in candidates:
            check_cancelled()
            for key in ('turn', 'gap'):
                for segment, edges in _fragments(chain, strengths, owners, ids, faces=key == 'turn'):
                    inner = segment[1:-1]
                    if not inner or any(owners.get(r['id']) not in (*ids, None) for r in inner):
                        continue
                    face_score = None
                    if key == 'gap':
                        if not any(owners.get(r['id']) is None for r in inner):
                            continue
                    else:
                        orientations = {quality(r).get('orientation') for r in segment} - {None, 'unknown'}
                        if len(orientations) < 2 or any(
                            r['_face_vector'] is None for r in (segment[0], segment[-1])
                        ):
                            continue
                        face_score = scores.face_similarity(
                            segment[0]['_face_vector'], segment[-1]['_face_vector']
                        )
                        if face_score < config.data['recognition']['match_threshold']:
                            continue
                    rank = (min(edges), -segment[0]['cue'])
                    if key not in best or rank > best[key][0]:
                        best[key] = rank, segment, edges, face_score
    frames = defaultdict(list)
    for row in rows:
        frames[row['cue']].append(row)
    for route in (turn, gap):
        entry = best.get(route['strategy'])
        if entry:
            _, segment, edges, face_score = entry
            _describe_chain(config, route, segment, edges, owners, face_score, frames, scores)
        else:
            route['notes'].append(
                'No physically separated same-shot body path with the required A/B endpoints.'
            )
    return [turn, crossing, gap]
