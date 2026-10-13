"""Rank all retained VTT observations; encode only the first usable winner."""
import hashlib
import heapq
import json
import uuid

from stash_ai.runtime.cleanup import closing
from .clustering import _groups, reference_context
from .config import plain
from .database import encoded
from .media import safe_url
from .models import checked_detection
from .portraits import policy_identity
from .quality import PortraitFramingError, portrait_bytes
from .portrait_availability import measure_available, score_metadata, observation_identity, retained_image, retained_portrait, cached_thumbnail
from .visibility import cached as cached_visibility, required as visibility_required
from .expression import cached as cached_expression, required as expression_required


class Sources:
    """One active scene source, including both visibility and winner extraction."""
    def __init__(self, client, config, factory):
        self.client, self.config, self.factory = client, config, factory
        self.scene = self.source = None
        self.failed = set()

    def thumbnail(self, scene, row):
        if scene in self.failed:
            raise ValueError('VTT source could not be verified.')
        if self.scene != scene:
            self.close()
            try:
                self.source = self.factory(self.client, self.client.scene(scene), self.config).__enter__()
                self.scene = scene
            except Exception:
                self.failed.add(scene)
                raise
        try:
            if self.source.fingerprint != row['input_sha']:
                raise ValueError('VTT source changed after recognition.')
            cue = self.source.cues[row['cue']]
            if (cue.start != row['seconds'] or safe_url(cue.image) != row['source']
                    or list(cue.rect) != json.loads(row['rect'])):
                raise ValueError('VTT cue does not match its recognition observation.')
            return self.source.thumbnail(row['cue'])
        except Exception:
            self.failed.add(scene)
            raise

    def close(self):
        if self.source is not None:
            self.source.close()
        self.scene = self.source = None


def store(database, identifier, scope, target, scene, metadata):
    with database.connection:
        existing = database.connection.execute('SELECT metadata,image FROM portraits WHERE id=?', (identifier,)).fetchone()
        if existing:
            # Retain any previously prepared JPEG, including one now ineligible.
            retained = json.loads(existing['metadata'])
            if existing['image'] is not None:
                # Re-ranking cannot relabel a cached recognition crop as a live
                # thumbnail, or change the coordinates of already encoded bytes.
                for name in ('source_type', 'crop', 'crop_coordinate_system', 'encoding', 'pixels',
                             'recognition_crop_bounds', 'fallback'):
                    if name in retained:
                        metadata[name] = retained[name]
            retained.update(metadata)
            if not metadata.get('reason'):
                retained.pop('reason', None)
            database.connection.execute('UPDATE portraits SET metadata=? WHERE id=?', (encoded(retained), identifier))
        else:
            database.connection.execute('INSERT INTO portraits VALUES (?,?,?,?,?,?,?,?)',
                (identifier, scope, target['person'], scene, encoded(metadata), None, None,
                 'skipped' if metadata.get('reason') else 'eligible'))


def choose_vtt(database, client, config, scope, target, source_factory, visibility, expression, *, fallback_reason=None):
    db, policy = database.connection, config.data['portraits']
    person = db.execute('SELECT * FROM people WHERE id=? AND scope=?', (target['person'], scope)).fetchone()
    groups = {group.identifier: group for group in _groups(db, scope, person['namespace'], config.data['clustering'])}
    context = reference_context(groups.values())
    policy_sha = policy_identity(config)
    ranked = []  # Only scores/IDs, never all embedding or JPEG BLOBs.
    with closing(Sources(client, config, source_factory), name='close VTT portrait source') as sources:
        for selected in sorted(target['scans'], key=lambda item: (item['scene'], item['scan'])):
            scene = selected['scene']
            rows = db.execute('''SELECT o.*,s.input_sha FROM observations o
              JOIN members m ON m.observation=o.id JOIN scans s ON s.id=o.scan
              WHERE m.person=? AND o.scan=? AND s.scene=? AND s.scope=?
              AND s.namespace=? AND s.status='complete' AND o.rejected IS NULL
              ORDER BY o.cue,o.id''', (target['person'], selected['scan'], scene, scope, person['namespace']))
            for row in rows:
                identifier = str(uuid.uuid5(uuid.NAMESPACE_URL,
                    encoded([scope, target['person'], row['id'], 'vtt', policy_sha])))
                quality = json.loads(row['quality'])
                metadata = {'version': 3, 'source_type': 'vtt', 'requested_source': policy['source'],
                    'selection_policy': 'best-available-vtt-observations-v3',
                    'policy_sha': policy_sha, 'policy': plain(policy), 'observation': row['id'],
                    'scan': row['scan'], 'scene': scene, 'person': target['person'],
                    'performer': target['performer'], 'namespace': person['namespace'],
                    'source': row['source'], 'rect': json.loads(row['rect']), 'cue': row['cue'],
                    'cue_seconds': row['seconds'], 'geometry': json.loads(row['geometry']),
                    'quality': quality, 'quality_errors': {}, 'reference_context': context, 'input_sha': row['input_sha']}
                if fallback_reason:
                    metadata['fallback'] = {'from': policy['source'], 'reason': fallback_reason}
                try:
                    reason, similarity, runner, basis = observation_identity(database, row, groups, target, config, context)
                    metadata.update(similarity=similarity, runner_up=runner, identity_basis=basis)
                except (TypeError, ValueError):
                    reason = 'portrait_identity_unverified'
                needs_visibility = visibility_required(policy) and not cached_visibility(quality)
                needs_expression = expression_required(policy) and not cached_expression(quality)
                if not reason and (needs_visibility or needs_expression):
                    try:
                        face = checked_detection(metadata['geometry'])
                        image = sources.thumbnail(scene, row)
                    except Exception:
                        metadata['source_warning'] = 'vtt_source_unavailable_or_changed'
                        try:
                            image, face = retained_image(row)
                            metadata['quality_source'] = 'recognition_crop'
                        except Exception:
                            image = None
                    if image is not None:
                        with image:
                            metadata['quality_errors'] = measure_available(client, image, face, quality, policy, visibility, expression)
                        # Independent of framing, thresholds, weights and selected performer.
                        with db:
                            db.execute('UPDATE observations SET quality=? WHERE id=?', (encoded(quality), row['id']))
                if reason:
                    metadata['reason'] = reason
                else:
                    score_metadata(metadata, quality, policy)
                    heapq.heappush(ranked, (-metadata['score'], -metadata['quality_tiebreak'], scene, row['seconds'], identifier))
                store(database, identifier, scope, target, scene, metadata)

        # All eligibility and ranking decisions precede JPEG preparation. A bad
        # crop does not hide a later valid observation behind a candidate limit.
        considered = len(ranked)
        fallbacks = []
        while ranked:
            _, _, scene, _, identifier = heapq.heappop(ranked)
            existing = db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
            metadata = json.loads(existing['metadata'])
            metadata['eligible_observations'] = considered
            if existing['image'] is not None and metadata['source_type'] == 'vtt':
                with db:
                    db.execute("UPDATE portraits SET metadata=?,status='selected' WHERE id=?", (encoded(metadata), identifier))
                return db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
            row = db.execute('SELECT o.*,s.input_sha FROM observations o JOIN scans s ON s.id=o.scan WHERE o.id=?',
                             (metadata['observation'],)).fetchone()
            reused = cached_thumbnail(database, config, scope, target, row, identifier, metadata)
            if reused is not None:
                return reused
            try:
                encoding = {}
                with sources.thumbnail(scene, row) as image:
                    data, bounds = portrait_bytes(image, checked_detection(metadata['geometry']), policy,
                                                 limit=config.data['storage']['max_crop_bytes'], details=encoding)
                metadata.update(source_type='vtt', crop=bounds, pixels=encoding['pixels'], encoding=encoding,
                                crop_coordinate_system='thumbnail')
                metadata.pop('source_warning', None)
                metadata.pop('recognition_crop_bounds', None)
                if not fallback_reason:
                    metadata.pop('fallback', None)
                else:
                    metadata['fallback'] = {'from': policy['source'], 'reason': fallback_reason}
            except PortraitFramingError:
                metadata['source_warning'] = 'portrait_encoding_unavailable'
                store(database, identifier, scope, target, scene, metadata)
                fallbacks.append(identifier)
                continue
            except Exception:
                metadata['source_warning'] = 'vtt_source_unavailable_or_changed'
                store(database, identifier, scope, target, scene, metadata)
                fallbacks.append(identifier)
                continue
            with db:
                db.execute("UPDATE portraits SET metadata=?,image=?,digest=?,status='selected' WHERE id=?",
                           (encoded(metadata), data, hashlib.sha256(data).hexdigest(), identifier))
            return db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
        # Prefer a real thumbnail whenever one is usable; retained recognition
        # pixels still illustrate the accepted identity if every source fails.
        for identifier in fallbacks:
            existing = db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
            metadata = json.loads(existing['metadata'])
            row = db.execute('SELECT * FROM observations WHERE id=?', (metadata['observation'],)).fetchone()
            try:
                data, metadata = retained_portrait(row, metadata, policy, config.data['storage']['max_crop_bytes'])
            except Exception:
                metadata['reason'] = 'portrait_source_unavailable'
                store(database, identifier, scope, target, existing['scene'], metadata)
                continue
            with db:
                db.execute("UPDATE portraits SET metadata=?,image=?,digest=?,status='selected' WHERE id=?",
                           (encoded(metadata), data, hashlib.sha256(data).hexdigest(), identifier))
            return db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
    return None
