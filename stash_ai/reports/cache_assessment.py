"""Assess retained evidence and scene-wide publication blockers."""
from dataclasses import dataclass, replace

from stash_ai.storage.results import KINDS

STATUSES = (
    'ready', 'rethreshold', 'missing', 'partial', 'refresh_required',
    'legacy_only', 'insufficient', 'inconsistent',
)
REUSABLE = frozenset(('ready', 'rethreshold'))
RESTART = frozenset(('legacy_only', 'insufficient', 'inconsistent'))


def inspect_output(app, config, kind, segments, index, refresh, checkpoint, error):
    from stash_ai.evidence.cache_views import scene_threshold, validate_scene
    from stash_ai.evidence.temporal.policy import marker_floor
    from stash_ai.evidence.thresholds import _score, indexed_signature, marker_threshold, validate_marker_blocklist

    def result(status, reason, **details):
        return dict(status=status, reason=reason, **details)

    if error:
        return result('inconsistent', error, unreadable=True)
    if refresh:
        return result('refresh_required', 'An outstanding refresh invalidates the saved result.')
    if segments is None:
        if index is not None:
            return result('inconsistent', 'Completion index exists without evidence; run --db-check.')
        return result('partial' if checkpoint else 'missing',
                      'Checkpoint exists; compatibility is checked when inference resumes.' if checkpoint
                      else 'No saved result for this output.')
    if not segments:
        return result('inconsistent', 'A result row has no segments; this is not completed zero-detection inference.')
    inference = next((segment for segment in reversed(segments)
                      if segment['meta'].get('origin') == 'inference'), None)
    if inference is None:
        return result('legacy_only', 'Imported output does not prove completed model inference.')
    signature = inference['meta'].get('inference_signature')
    expected = (1, int(any(segment['populated'] for segment in segments)), indexed_signature(inference))
    if not isinstance(signature, str) or not signature or index != expected:
        return result('inconsistent', 'Evidence and completion index disagree; run --db-check.')

    threshold = config.scene_threshold if kind == 'scene' else config.marker_threshold
    needed = (threshold * app.TEMPORAL_POLICY.scene_support_ratio
              if kind == 'scene' else marker_floor(app.TEMPORAL_POLICY, threshold))
    requested = getattr(config, 'scene_cache_floor' if kind == 'scene' else 'marker_cache_floor')
    required_floor = min(needed, requested) if requested is not None else needed
    floors, effective = [], []
    try:
        for segment in segments:
            meta = segment['meta']
            if kind == 'scene':
                validate_scene(app, segment, threshold, has_frames=segment['has_frames'])
                floor, saved = meta['scene_score_floor'], scene_threshold(segment)
            else:
                if segment['mode'] != 'frames' or meta.get('origin') != 'inference':
                    raise ValueError('Imported marker ranges cannot prove retained frame coverage.')
                validate_marker_blocklist(segment, config.source.key)
                floor = meta.get('marker_score_floor')
                saved = marker_threshold(segment)
            if not _score(saved) or not _score(floor):
                raise ValueError('Invalid saved threshold or retention floor; run --db-check.')
            if floor > required_floor:
                raise ValueError(f'Retained floor {floor:g} exceeds required {required_floor:g}.')
            floors.append(floor)
            effective.append(saved)
    except (KeyError, ValueError) as exc:
        return result('insufficient', str(exc), required_floor=required_floor,
                      target_threshold=threshold)
    changed = any(value != threshold for value in effective)
    return result('rethreshold' if changed else 'ready',
                  'Retained evidence supports YAML thresholds; apply rethreshold without inference.' if changed
                  else 'Completed evidence supports the selected thresholds and retention.',
                  retained_floor_max=max(floors), required_floor=required_floor,
                  target_threshold=threshold, saved_thresholds=sorted(set(effective)))


def publication_dependencies(app, evidence, selected):
    """Merge reads unselected cached models at their saved effective thresholds."""
    from stash_ai.evidence.cache_views import scene_threshold, validate_scene
    from stash_ai.models import SOURCES
    from stash_ai.evidence.thresholds import validate_marker_cache

    dependencies = []
    context = replace(app, MARKER_THRESHOLDS=None, SCENE_THRESHOLDS=None)
    for (model, kind), (segments, error) in sorted(evidence.items()):
        if model in selected:
            continue
        try:
            if error:
                raise ValueError(error)
            if model not in SOURCES:
                raise ValueError('Unknown cached model; run --db-check.')
            if kind == 'markers':
                if any(segment['mode'] not in ('frames', 'spans') for segment in segments):
                    raise ValueError('Unsupported stored marker evidence; run --db-check.')
                validate_marker_cache(context, {model: segments})
            else:
                for segment in segments:
                    meta = segment['meta']
                    if meta.get('origin') != 'inference':
                        continue
                    threshold = scene_threshold(segment)
                    validate_scene(app, segment, threshold, has_frames=segment['has_frames'])
        except (KeyError, TypeError, ValueError) as exc:
            dependencies.append(dict(model=model, kind=kind, reason=str(exc),
                resolution=f'Include {model} with --models and regenerate the report; '
                           'resolve unreadable or invalid evidence with --db-check first.'))
    return dependencies


@dataclass
class PairAssessment:
    scene_id: str
    model: str
    outputs: dict
    blockers: list[str]
    checkpoint: dict | None
    publication_dependencies: list[dict]
    blocking_models: list[str]


def assess_scene(app, scene, settings, evidence, indexes, refreshes, checkpoints,
                 *, cleanup_pending, publication_pending, outside_refresh):
    """Assess all selected models before propagating scene-wide blockers."""
    models = [config.source.key for config in settings]
    dependencies = publication_dependencies(app, evidence, models)
    inspected = []
    for config in settings:
        model = config.source.key
        checkpoint = checkpoints.get((scene, model))
        outputs = {}
        for kind in KINDS:
            segments, error = evidence.get((model, kind), (None, None))
            outputs[kind] = inspect_output(app, config, kind, segments,
                indexes.get((scene, model, kind)), (scene, model, kind) in refreshes,
                checkpoint, error)
        blockers = []
        if cleanup_pending:
            blockers.append('pending_global_cleanup')
        if publication_pending:
            blockers.append('pending_publication')
        if outside_refresh:
            blockers.append('refresh_required_for_unselected_model')
        if dependencies:
            blockers.append('unusable_evidence_for_unselected_model')
        if any(value.get('unreadable') for value in outputs.values()):
            blockers.append('unreadable_evidence_run_db_check')
        inspected.append(PairAssessment(scene, model, outputs, blockers, checkpoint, dependencies, []))
    # Every planned command attempts publication, so selected peers must all
    # inherit a blocked model's scene restriction, independent of model order.
    blocking_models = sorted(pair.model for pair in inspected if pair.blockers)
    for pair in inspected:
        if blocking_models and not pair.blockers:
            pair.blockers.append('blocked_selected_model')
            pair.blocking_models = blocking_models
    return inspected
