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
from .diagnostics import metadata as safe_metadata
from .models import checked_detection
from .portraits import policy_identity
from .quality import PortraitFramingError, portrait_bytes
from .portrait_availability import measure_available, score_metadata, observation_identity, retained_image, retained_portrait, cached_thumbnail, fit_cached_bytes, measurement_basis
from .visibility import cached as cached_visibility, required as visibility_required
from .expression import cached as cached_expression, required as expression_required
from .progress import tracking
from .portrait_availability import candidate_observations, crop_row
from .clustering import matching_policy
from .portrait_writes import execute as cache_write, flush, batched
from .vtt_ranking_cache import Ranking


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
                # A later candidate can retry a transient acquisition failure.
                self.scene = self.source = None
                raise
        try:
            if self.source.fingerprint != row['input_sha']:
                raise ValueError('VTT source changed after recognition.')
            cue = self.source.cues[row['cue']]
            if (cue.start != row['seconds'] or safe_url(cue.image) != safe_url(row['source'])
                    or list(cue.rect) != json.loads(row['rect'])):
                raise ValueError('VTT cue does not match its recognition observation.')
        except (ValueError, IndexError, KeyError, TypeError):
            self.failed.add(scene)
            raise
        return self.source.thumbnail(row['cue'])

    def close(self):
        source = self.source
        self.scene = self.source = None
        if source is not None:
            source.close()


def store(database, identifier, scope, target, scene, metadata):
    existing = database.connection.execute('SELECT metadata,image IS NOT NULL AS has_image '
                                           'FROM portraits WHERE id=?', (identifier,)).fetchone()
    if existing:
        # Retain any previously prepared JPEG, including one now ineligible.
        try:
            retained = json.loads(existing['metadata'])
            if not isinstance(retained,dict):
                raise ValueError('Cached portrait metadata is not an object.')
        except (ValueError,TypeError):
            retained = {}
            if existing['has_image']:
                metadata['source_type'] = 'unverified_cache'
        if existing['has_image']:
            # Re-ranking cannot relabel a cached recognition crop as a live
            # thumbnail, or change the coordinates of already encoded bytes.
            for name in ('source_type', 'crop', 'crop_coordinate_system', 'encoding', 'pixels',
                         'recognition_crop_bounds', 'fallback'):
                if name in retained:
                    metadata[name] = retained[name]
        retained.update(metadata)
        if not metadata.get('reason'):
            retained.pop('reason', None)
        payload = encoded(safe_metadata(retained))
        if payload != existing['metadata']:
            cache_write(database.connection,'UPDATE portraits SET metadata=? WHERE id=?', (payload, identifier))
    else:
        cache_write(database.connection,'INSERT INTO portraits VALUES (?,?,?,?,?,?,?,?)',
            (identifier, scope, target['person'], scene, encoded(metadata), None, None,
             'skipped' if metadata.get('reason') else 'eligible'))


@batched
def choose_vtt(database, client, config, scope, target, source_factory, visibility, expression, *, fallback_reason=None):
    db, policy = database.connection, config.data['portraits']
    person = db.execute('SELECT * FROM people WHERE id=? AND scope=?', (target['person'], scope)).fetchone()
    from .context import portrait_references
    groups, context = portrait_references(database, scope, person['namespace'], config,
        lambda: _groups(db, scope, person['namespace'], config.data['clustering']))
    policy_sha = policy_identity(config)
    ranking=Ranking(db,config,scope,target,person['namespace'],context,policy_sha,
                    enabled=policy['source']=='vtt' and fallback_reason is None)
    cached=ranking.load()
    ranked=cached if cached is not None else []
    def finish(result):
        flush(db)
        ranking.save(all_ranks)
        return result
    with closing(Sources(client, config, source_factory), name='close VTT portrait source') as sources:
        if cached is None:
            for selected in sorted(target['scans'], key=lambda item: (item['scene'], item['scan'])):
                scene = selected['scene']
                rows = db.execute('''SELECT o.id,o.scan,o.cue,o.seconds,o.source,o.rect,o.geometry,o.quality,
                  o.dimensions,o.embedding,s.input_sha FROM observations o
                  JOIN members m ON m.observation=o.id JOIN scans s ON s.id=o.scan
                  WHERE m.person=? AND o.scan=? AND s.scene=? AND s.scope=?
                  AND s.namespace=? AND s.status='complete' AND o.rejected IS NULL
                  ORDER BY o.cue,o.id''', (target['person'], selected['scan'], scene, scope, person['namespace']))
                with tracking(
                    candidate_observations(db, rows, selected['scan'], matching_policy(config)),
                    'Rank VTT portrait observations',
                    unit='observation',
                    total=lambda: db.execute(
                        'SELECT COUNT(*) FROM observations o JOIN members m ON m.observation=o.id JOIN '
                        'scans s ON s.id=o.scan WHERE m.person=? AND o.scan=? AND s.scene=? AND s.scope=?'
                        " AND s.namespace=? AND s.status='complete' AND o.rejected IS NULL",
                        (target['person'], selected['scan'], scene, scope, person['namespace']),
                    ).fetchone()[0],
                ) as tracked:
                    for row, candidate in tracked:
                        identifier = str(uuid.uuid5(uuid.NAMESPACE_URL,
                            encoded([scope, target['person'], row['id'], 'vtt', policy_sha])))
                        quality = json.loads(row['quality'])
                        original_quality = encoded(quality)
                        metadata = {'version': 3, 'source_type': 'vtt', 'requested_source': policy['source'],
                            'selection_policy': 'best-available-vtt-observations-v3',
                            'policy_sha': policy_sha, 'policy': plain(policy), 'observation': row['id'],
                            'scan': row['scan'], 'scene': scene, 'person': target['person'],
                            'performer': target['performer'], 'namespace': person['namespace'],
                            'source': safe_url(row['source']), 'rect': json.loads(row['rect']), 'cue': row['cue'],
                            'cue_seconds': row['seconds'], 'geometry': json.loads(row['geometry']),
                            'quality': quality, 'quality_errors': {}, 'reference_context': context, 'input_sha': row['input_sha']}
                        if fallback_reason:
                            metadata['fallback'] = {'from': policy['source'], 'reason': fallback_reason}
                        try:
                            reason, similarity, runner, basis = observation_identity(database, row, groups, target, config, context,
                                                                                    candidate=candidate)
                            metadata.update(similarity=similarity, runner_up=runner, identity_basis=basis)
                        except (TypeError, ValueError):
                            reason = 'portrait_identity_unverified'
                        required = {name for name, needed in (
                            ('visibility', visibility_required(policy)), ('expression', expression_required(policy))) if needed}
                        cached = {'visibility': cached_visibility, 'expression': cached_expression}
                        bases = quality.setdefault('measurement_basis', {})
                        # Older fallback caches recorded provenance only on portraits.
                        # Recover it before trusting their model-contract-only cache.
                        legacy_crop = db.execute("SELECT 1 FROM portraits WHERE scope=? AND person=? AND json_valid(metadata) "
                            "AND json_extract(metadata,'$.observation')=? "
                            "AND json_extract(metadata,'$.quality_source')='recognition_crop' LIMIT 1",
                            (scope, target['person'], row['id'])).fetchone()
                        for name in required:
                            if cached[name](quality) and name not in bases:
                                bases[name] = measurement_basis(crop_row(db, row) if legacy_crop else row,
                                                                'recognition_crop' if legacy_crop else 'vtt')
                        missing = {name for name in required if not cached[name](quality)}
                        approximate = {name for name in required if bases.get(name, {}).get('source_type') == 'recognition_crop'}
                        if not reason and (missing or approximate):
                            source_type = 'vtt'
                            try:
                                face = checked_detection(metadata['geometry'])
                                image = sources.thumbnail(scene, row)
                            except Exception:
                                metadata['source_warning'] = 'vtt_source_unavailable_or_changed'
                                source_type = 'recognition_crop'
                                try:
                                    image, face = retained_image(crop_row(db, row))
                                except Exception:
                                    image = None
                            if image is not None:
                                force = approximate if source_type == 'vtt' else set()
                                requested = missing | force
                                with image:
                                    metadata['quality_errors'] = measure_available(client, image, face, quality, policy,
                                        visibility, expression, force=force)
                                for name in requested:
                                    if name not in metadata['quality_errors'] and cached[name](quality):
                                        bases[name] = measurement_basis(crop_row(db, row) if source_type == 'recognition_crop' else row,
                                                                        source_type)
                        # Reuse remains independent of framing and ranking preferences.
                        if encoded(quality) != original_quality:
                            cache_write(db,'UPDATE observations SET quality=? WHERE id=?', (encoded(quality), row['id']))
                        source_types = {basis['source_type'] for name, basis in bases.items() if name in required}
                        if source_types:
                            metadata['quality_source'] = next(iter(source_types)) if len(source_types) == 1 else 'mixed'
                        if reason:
                            metadata['reason'] = reason
                        else:
                            score_metadata(metadata, quality, policy)
                            ranked.append((-metadata['score'], -metadata['quality_tiebreak'], scene, row['seconds'], identifier))
                        store(database, identifier, scope, target, scene, metadata)

        # All eligibility and ranking decisions precede JPEG preparation. A bad
        # crop does not hide a later valid observation behind a candidate limit.
        flush(db)
        all_ranks=list(ranked)
        heapq.heapify(ranked)
        considered = len(ranked)
        fallbacks = []
        def candidates():
            while ranked:
                yield heapq.heappop(ranked)
        with tracking(
            candidates(),
            'Choose VTT portrait',
            total=considered,
            unit='candidate',
        ) as choices:
            for _, _, scene, _, identifier in choices:
                existing = db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
                metadata = json.loads(existing['metadata'])
                metadata['eligible_observations'] = considered
                if existing['image'] is not None and metadata['source_type'] == 'vtt':
                    try:
                        data, sha = fit_cached_bytes(existing['image'], existing['digest'], config, metadata)
                    except (OSError, ValueError):
                        pass  # Try verified old/source pixels, preserving corrupt historical bytes.
                    else:
                        with db:
                            db.execute("UPDATE portraits SET metadata=?,image=?,digest=?,status='selected' WHERE id=?",
                                       (encoded(metadata), data, sha, identifier))
                        return finish(db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone())
                row = db.execute('SELECT o.*,s.input_sha FROM observations o JOIN scans s ON s.id=o.scan WHERE o.id=?',
                                 (metadata['observation'],)).fetchone()
                reused = cached_thumbnail(database, config, scope, target, row, identifier, metadata)
                if reused is not None:
                    return finish(reused)
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
                return finish(db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone())
        # Prefer a real thumbnail whenever one is usable; retained recognition
        # pixels still illustrate the accepted identity if every source fails.
        with tracking(
            fallbacks,
            'Check retained portrait fallbacks',
            unit='candidate',
        ) as tracked:
            for identifier in tracked:
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
                return finish(db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone())
    return finish(None)
