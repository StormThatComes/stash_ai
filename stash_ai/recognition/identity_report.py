"""Offline export of all current identity members and their recorded decisions.

Read one consistent SQLite snapshot. Never infer assignments from present-day
similarities, fetch media, load models, or modify recognition/Stash state.
"""
from collections import Counter, defaultdict
from contextlib import closing
from io import BytesIO
import json
from pathlib import Path

from .alignment_audit import _png, _private_directory, _write_json
from .clustering import matching_policy
from .config import plain
from .fragmentation_report import _open_readonly
from .models import namespace
from .pose_policy import is_anchor
from .scanning import scan_identity


from .decision_evidence import REASONS, DECISION_FIELDS, recorded, explanation

QUALITY_FIELDS = (
    'pixels', 'pose_deviation', 'roll_deviation', 'pose_contract', 'sharpness', 'exposure', 'clipped_fraction',
    'visible_face_fraction', 'landmarks_inside', 'complete', 'alignment_valid',
    'crop_bounds', 'crop_encoding',
)


def _object(raw):
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Invalid retained identity report metadata; expected a JSON object.')
    return value


def _select(db, model_namespace, identity_id, performer_id, scope):
    if (identity_id is None) == (performer_id is None):
        raise ValueError('Select exactly one --identity-id or --performer-id.')
    column, value = ('id', identity_id) if identity_id is not None else ('performer', performer_id)
    rows = db.execute(f'SELECT id,scope,namespace,created,performer FROM people WHERE {column}=? AND namespace=? '
                      'AND (? IS NULL OR scope=?) ORDER BY scope,id',
                      (str(value), model_namespace, scope, scope)).fetchall()
    if not rows:
        raise ValueError('No identity found for this selector, recognizer namespace and scope.')
    if len(rows) != 1:
        raise ValueError('Identity selector is ambiguous; specify --scope or --identity-id.')
    requested = person = dict(rows[0])
    seen = set()
    while True:
        if person['id'] in seen:
            raise ValueError('Identity merge history contains a cycle.')
        seen.add(person['id'])
        merge = db.execute('SELECT target FROM identity_merges WHERE source=?', (person['id'],)).fetchone()
        if merge is None:
            return requested, person
        row = db.execute('SELECT id,scope,namespace,created,performer FROM people WHERE id=?',
                         (merge['target'],)).fetchone()
        if row is None or row['scope'] != requested['scope'] or row['namespace'] != model_namespace:
            raise ValueError('Identity merge history crosses scope/namespace or has a missing target.')
        person = dict(row)


def _merges(db, person):
    rows = db.execute(
        'WITH RECURSIVE family(id) AS (SELECT ? UNION SELECT m.source FROM identity_merges m '
        'JOIN family f ON m.target=f.id) '
        'SELECT m.* FROM identity_merges m JOIN family f ON m.source=f.id ORDER BY m.created,m.source',
        (person['id'],)).fetchall()
    identifiers = {person['id']} | {row['source'] for row in rows}
    for identifier in identifiers:
        row = db.execute('SELECT scope,namespace FROM people WHERE id=?', (identifier,)).fetchone()
        if row is None or (row['scope'], row['namespace']) != (person['scope'], person['namespace']):
            raise ValueError('Identity merge history contains incompatible people.')
    return identifiers, [dict(source=row['source'], target=row['target'], created=row['created'],
                               status=row['status'], evidence=_object(row['evidence'])) for row in rows]


def _decisions(db, person, family, policy):
    result = defaultdict(list)
    # One metadata pass over this person's members; avoid a full candidate-table
    # lookup for every exported image. Historical policies remain separate.
    rows = db.execute(
        "SELECT c.* FROM candidates c JOIN observations o ON o.scan=c.scan "
        "AND o.id=json_extract(c.payload,'$.observation') "
        'JOIN members m ON m.observation=o.id WHERE m.person=? ORDER BY c.policy,c.id',
        (person['id'],))
    for row in rows:
        payload = _object(row['payload'])
        result[payload['observation']].append({
            'candidate_id': row['id'], 'policy': row['policy'],
            'matches_current_policy': row['policy'] == policy,
            'person_id': row['person'], 'status': row['status'],
            'supports_membership': row['status'] == 'accepted' and row['person'] in family,
            'recorded': recorded(payload),
            'explanation': explanation(payload),
        })
    return result


def _assignment(decisions):
    supporting = [item for item in decisions if item['supports_membership']]
    current = [item for item in supporting if item['matches_current_policy']]
    selected = current or supporting
    categories = {REASONS.get(item['recorded'].get('reason'), ('unknown', ''))[0] for item in selected}
    category = next(iter(categories)) if len(categories) == 1 else 'unknown'
    return {
        'category': category,
        'role': ('anchor' if category.startswith('anchor') else
                 'auxiliary' if category.startswith('auxiliary') else
                 'attachment' if category == 'hybrid-attachment' else 'unknown'),
        'evidence_status': ('recorded_current_policy' if current else
                            'recorded_other_policy' if supporting else 'membership_only'),
        'reason_codes': sorted({item['recorded']['reason'] for item in selected if item['recorded'].get('reason')}),
        'explanation': ('The members table assigns this observation to the reported identity. '
                        'The decisions below are recorded evidence, not a reconstruction of the original matching run.'
                        if selected else
                        'The members table assigns this observation to the reported identity, but no accepted '
                        'decision for this identity or a merged predecessor is retained; the reason is unknown.'),
    }


def _image(blob, path, *, annotate=None):
    from PIL import Image, UnidentifiedImageError
    if blob is None:
        return {'status': 'missing_crop', 'file': None}
    try:
        with Image.open(BytesIO(blob)) as original:
            rgb = original.convert('RGB')
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        return {'status': 'invalid_crop', 'file': None}
    # Keep writes outside the decode error handler: disk/permission failures
    # must abort the export rather than masquerade as invalid retained pixels.
    with rgb:
        pixels = list(rgb.size)
        _png(rgb, path)
        annotations = annotate(rgb, path) if annotate is not None else None
    return {'status': 'exported', 'file': path.name, 'pixels': pixels,
            'source': 'retained_recognition_crop',
            **({'annotations': annotations} if annotations is not None else {})}


def export(config, output_dir, *, identity_id=None, performer_id=None, scope=None):
    """Export every membership, never capped by the recognition reference limit."""
    directory = Path(output_dir).expanduser()
    if directory.exists():
        raise FileExistsError(f'Identity report output directory already exists: {directory}')
    policy, model_namespace = matching_policy(config), namespace(config)
    with closing(_open_readonly(config.database)) as db:
        requested, person = _select(db, model_namespace, identity_id, performer_id, scope)
        family, merges = _merges(db, person)
        invalid = db.execute('SELECT 1 FROM members m JOIN observations o ON o.id=m.observation '
            'JOIN scans s ON s.id=o.scan WHERE m.person=? AND (s.scope<>? OR s.namespace<>?) LIMIT 1',
            (person['id'], person['scope'], model_namespace)).fetchone()
        if invalid:
            raise ValueError('Identity has incompatible retained membership evidence.')
        decisions = _decisions(db, person, family, policy)
        origins = defaultdict(list)
        for row in db.execute('SELECT i.observation,i.person FROM identity_origins i '
                              'JOIN members m ON m.observation=i.observation WHERE m.person=? '
                              'ORDER BY i.observation,i.person', (person['id'],)):
            origins[row['observation']].append(row['person'])
        report = {
            'format': 1, 'kind': 'recognition_identity',
            'requested_identity': requested, 'identity': person,
            'current_configuration': {
                'recognizer': config.data['models']['recognizer'], 'matching_policy': policy,
                **{key: plain(config.data[key]) for key in ('detection', 'recognition', 'clustering')},
            },
            'cue_index_base': 0, 'identity_merges': merges,
            'limitations': [
                'All current members are included without detection filtering or a sample cap; this is not a rescan.',
                'PNGs decode retained recognition crops (normally JPEG), not original thumbnails or aligned model inputs.',
                'Annotated copies show retained source landmarks mapped to crop pixels; missing geometry is not inferred.',
                'Missing or unreadable crops cannot be recovered offline and remain explicit entries.',
                'Recorded decisions may be absent, from other policies, or overwritten by later rechecks/rebuilds.',
                'Scores are cosine similarities, not probabilities. Historical reference members, individual cohesion '
                'scores and runner-up identity IDs are not always retained; none are inferred here.',
                'Current anchor/reference eligibility describes this snapshot under the supplied configuration, '
                'not the original matching run. Original identity IDs are historical labels, not matching proof.',
            ],
            'samples': [],
        }
        rows = db.execute(
            'SELECT o.id,o.cue,o.seconds,o.rect,o.geometry,o.quality,o.rejected,o.crop,'
            's.id AS scan_id,s.scene,s.status AS scan_status,s.policy_sha AS detection_policy '
            'FROM observations o JOIN members m ON m.observation=o.id JOIN scans s ON s.id=o.scan '
            'WHERE m.person=? ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.geometry,o.embedding,o.id',
            (person['id'],))
        directory = _private_directory(directory)
        counts, reference_count = Counter(), 0
        detection_policy = scan_identity(config, model_namespace)
        for index, row in enumerate(rows, 1):
            scene, cue = row['scene'], row['cue']
            if (not scene.isascii() or not scene.isdecimal() or int(scene) <= 0
                    or type(cue) is not int or cue < 0):
                raise ValueError('Invalid retained scene/cue identifier.')
            quality = _object(row['quality'])
            recorded = decisions.get(row['id'], [])
            assignment = _assignment(recorded)
            name = f'scene-{int(scene)}_cue-{cue:06d}_{assignment["category"]}_face-{index:06d}.png'
            geometry = json.loads(row['geometry'])
            from .retained_annotations import export as annotate
            image = _image(row['crop'], directory / name,
                           annotate=lambda rgb, path: annotate(rgb, path, geometry, quality.get('crop_bounds')))
            eligible = assignment['category'] != 'hybrid-attachment' and row['rejected'] is None and row['scan_status'] == 'complete' and is_anchor(quality, config.data['clustering'])
            reference_count += int(eligible)
            sample = {
                'observation_id': row['id'], 'scene_id': scene, 'scan_id': row['scan_id'],
                'scan_status': row['scan_status'], 'cue_index': cue, 'seconds': row['seconds'],
                'detection_policy': row['detection_policy'],
                'detection_policy_matches_current': row['detection_policy'] == detection_policy,
                'retained_rejection': row['rejected'],
                'quality': {key: quality[key] for key in QUALITY_FIELDS if key in quality},
                'geometry': geometry, 'source_rect': json.loads(row['rect']),
                'assignment': assignment, 'decisions': recorded,
                'original_identity_ids': origins.get(row['id'], []),
                'current_anchor_eligible': eligible,
                'current_reference_selected': eligible and reference_count <= config.data['clustering']['max_samples_per_person'],
                'image': image,
            }
            from .person_collection import retained_report
            sample['person_evidence'] = retained_report(db,row['id'],config,directory,name,geometry=geometry)
            report['samples'].append(sample)
            counts[image['status']] += 1
            counts[assignment['role']] += 1
        report['summary'] = {
            'members': len(report['samples']), 'pngs_exported': counts['exported'],
            'missing_crops': counts['missing_crop'], 'invalid_crops': counts['invalid_crop'],
            'roles': {role: counts[role] for role in ('anchor', 'auxiliary', 'unknown')+
                      (('attachment',) if counts['attachment'] else ())},
            'complete': counts['exported'] == len(report['samples']),
        }
        _write_json(directory / 'manifest.json', report)
        return report


def execute(args, config):
    result = export(config, args.output_dir, identity_id=args.identity_id,
                    performer_id=args.performer_id, scope=args.scope)
    summary = result['summary']
    print(f'Identity report written: {Path(args.output_dir).expanduser() / "manifest.json"} '
          f'({summary["pngs_exported"]}/{summary["members"]} face PNGs)')
    return 0 if summary['complete'] else 1
