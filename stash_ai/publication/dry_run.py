"""Stream concrete plans with symbolic references for tags not created yet."""
import json
from pathlib import Path
from stash_ai.runtime.status import completion_status
from stash_ai.runtime.cleanup import cleanup


class DryRunPlan:
    def __init__(self, destination):
        self.destination, self.stream = destination, None

    def __enter__(self):
        if self.destination is not None:
            destination = Path(self.destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.stream = destination.open('x', encoding='utf-8', newline='\n')
            try:
                self.write(dict(type='header', format=1, dry_run=True,
                                symbolic_id_prefix='dry-run:'))
            except BaseException:
                cleanup([self.close])
                raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stream is not None:
            callbacks = []
            if exc_type is not None:
                callbacks.append(lambda: self.write(dict(type=completion_status(exc_type),
                                                         error_type=exc_type.__name__)))
            cleanup(callbacks + [self.close], primary=exc_value)

    def close(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            stream.close()

    def write(self, value):
        if self.stream is not None:
            self.stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                         allow_nan=False, separators=(',', ':')) + '\n')
            self.stream.flush()

    def scene(self, scene_id, plan, resolver, app, stats):
        names = resolver.names_by_id
        referenced = (set(plan.final_scene_tag_ids) | {str(payload[field])
            for payload in [*plan.creates, *plan.updates] for field in ('primary_tag_id',)} |
            {str(tag) for payload in [*plan.creates, *plan.updates] for tag in payload['tag_ids']})
        self.write(dict(type='scene', scene_id=str(scene_id),
            scene_tags=dict(changed=bool(app.APPLY_SCENE_TAGS and plan.scene_changed),
                before=sorted(plan.existing_scene_tag_ids), after=sorted(plan.final_scene_tag_ids
                    if app.APPLY_SCENE_TAGS else plan.existing_scene_tag_ids)),
            creates=[dict(payload=payload, reason='no_owned_match') for payload in plan.creates],
            updates=[dict(payload=payload, reason='owned_payload_changed_or_excluded_extra') for payload in plan.updates],
            deletes=[dict(id=identifier, reason='obsolete_owned_marker_or_excluded_primary')
                     for identifier in sorted(plan.deletes)],
            tag_references={identifier: names.get(identifier) for identifier in sorted(referenced)},
            counts=dict(stats)))
