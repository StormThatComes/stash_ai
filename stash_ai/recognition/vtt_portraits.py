"""Rank all retained VTT observations; encode only the first usable winner."""
import hashlib
import heapq
import json
import uuid

from stash_ai.runtime.cleanup import closing
from .clustering import _groups, vector, reference_context
from .config import plain
from .database import encoded
from .media import safe_url
from .models import checked_detection
from .portraits import identity_reason, policy_identity
from .quality import PortraitFramingError, portrait_bytes, portrait_score, rejection
from .visibility import cached as cached_visibility, required as visibility_required


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
        existing = database.connection.execute('SELECT metadata FROM portraits WHERE id=?', (identifier,)).fetchone()
        if existing:
            # Retain any previously prepared JPEG, including one now ineligible.
            retained = json.loads(existing['metadata'])
            retained.update(metadata)
            if not metadata.get('reason'):
                retained.pop('reason', None)
            database.connection.execute('UPDATE portraits SET metadata=? WHERE id=?', (encoded(retained), identifier))
        else:
            database.connection.execute('INSERT INTO portraits VALUES (?,?,?,?,?,?,?,?)',
                (identifier, scope, target['person'], scene, encoded(metadata), None, None,
                 'skipped' if metadata.get('reason') else 'eligible'))


def choose_vtt(database, client, config, scope, target, source_factory, visibility):
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
                metadata = {'version': 2, 'source_type': 'vtt',
                    'selection_policy': 'all-vtt-observations-identity-first-v2',
                    'policy_sha': policy_sha, 'policy': plain(policy), 'observation': row['id'],
                    'scan': row['scan'], 'scene': scene, 'person': target['person'],
                    'performer': target['performer'], 'namespace': person['namespace'],
                    'source': row['source'], 'rect': json.loads(row['rect']), 'cue': row['cue'],
                    'cue_seconds': row['seconds'], 'geometry': json.loads(row['geometry']),
                    'quality': quality, 'reference_context': context, 'input_sha': row['input_sha']}
                reason = rejection(quality, policy, portrait=True, visibility=False)
                if not reason:
                    try:
                        reason, similarity, runner = identity_reason(vector(row), groups, target['person'], config)
                        metadata.update(similarity=similarity, runner_up=runner)
                    except (TypeError, ValueError):
                        reason = 'portrait_identity_unverified'
                if not reason and visibility_required(policy):
                    if not cached_visibility(quality):
                        try:
                            image = sources.thumbnail(scene, row)
                        except Exception:
                            reason = 'vtt_source_unavailable_or_changed'
                        else:
                            with image:
                                quality.update(visibility.measure(image, checked_detection(metadata['geometry'])))
                            # Independent of framing, thresholds, weights and selected performer.
                            with db:
                                db.execute('UPDATE observations SET quality=? WHERE id=?', (encoded(quality), row['id']))
                    if not reason:
                        reason = rejection(quality, policy, portrait=True)
                if reason:
                    metadata['reason'] = reason
                else:
                    metadata['score'] = portrait_score(quality, policy)
                    heapq.heappush(ranked, (-metadata['score'], scene, row['seconds'], identifier))
                store(database, identifier, scope, target, scene, metadata)

        # All eligibility and ranking decisions precede JPEG preparation. A bad
        # crop does not hide a later valid observation behind a candidate limit.
        considered = len(ranked)
        while ranked:
            _, scene, _, identifier = heapq.heappop(ranked)
            existing = db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
            metadata = json.loads(existing['metadata'])
            metadata['eligible_observations'] = considered
            if existing['image'] is not None:
                with db:
                    db.execute("UPDATE portraits SET metadata=?,status='selected' WHERE id=?", (encoded(metadata), identifier))
                return db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
            row = db.execute('SELECT o.*,s.input_sha FROM observations o JOIN scans s ON s.id=o.scan WHERE o.id=?',
                             (metadata['observation'],)).fetchone()
            try:
                with sources.thumbnail(scene, row) as image:
                    data, bounds = portrait_bytes(image, checked_detection(metadata['geometry']), policy,
                                                 limit=config.data['storage']['max_crop_bytes'])
                metadata.update(crop=bounds, pixels=[bounds[2] - bounds[0], bounds[3] - bounds[1]])
            except PortraitFramingError:
                metadata['reason'] = 'portrait_framing_ineligible'
                store(database, identifier, scope, target, scene, metadata)
                continue
            except Exception:
                metadata['reason'] = 'vtt_source_unavailable_or_changed'
                store(database, identifier, scope, target, scene, metadata)
                continue
            with db:
                db.execute("UPDATE portraits SET metadata=?,image=?,digest=?,status='selected' WHERE id=?",
                           (encoded(metadata), data, hashlib.sha256(data).hexdigest(), identifier))
            return db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
    return None
