"""Deduplicated review counts, model attribution and bounded examples."""
from collections import Counter

from stash_ai.models import SOURCES


class ReviewAccumulator:
    def __init__(self, selected_scenes, examples_limit, reasons):
        self.examples_limit = examples_limit
        self.reasons = tuple(reasons)
        self.inspected = {model: set() for model in SOURCES}
        self.flagged = {model: set() for model in SOURCES}
        self.model_counts = {model: Counter() for model in SOURCES}
        self.model_reasons = {model: Counter() for model in SOURCES}
        self.tags = {}
        self.skipped = []
        self.skip_counts = Counter()
        self.totals = Counter(selected_scenes=selected_scenes)
        self.all_tags = set()
        self.review_tags = set()

    def skip(self, scene, reason, message=None):
        self.skip_counts[reason] += 1
        if len(self.skipped) < 20:
            self.skipped.append(dict(scene_id=scene, reason=reason,
                                     **({'message': message} if message else {})))

    def record(self, scene, rows):
        """Rows already combine history and signal kinds for each model/tag."""
        self.totals['analyzed_scenes'] += 1
        flagged_models, flagged_labels = set(), set()
        labels = {tag for _, tag in rows}
        self.all_tags.update(labels)
        self.totals['scene_tag_pairs_reviewed'] += len(labels)
        for (model, tag), row in rows.items():
            self.inspected[model].add(tag)
            self.model_counts[model]['scene_tag_pairs_reviewed'] += 1
            if not row['reasons']:
                continue
            self.flagged[model].add(tag)
            flagged_models.add(model)
            flagged_labels.add(tag)
            self.model_counts[model]['scene_tag_pairs_flagged'] += 1
            self.model_reasons[model].update(row['reasons'])
            entry = self.tags.setdefault((model, tag), dict(
                model=model, tag='ai:' + tag, scene_count=0, reasons=Counter(), examples=[]))
            entry['scene_count'] += 1
            entry['reasons'].update(row['reasons'])
            if len(entry['examples']) < self.examples_limit:
                entry['examples'].append(dict(scene_id=scene, reasons=sorted(row['reasons']),
                                             evidence=row['evidence']))
        for model in flagged_models:
            self.model_counts[model]['scenes_flagged'] += 1
        self.review_tags.update(flagged_labels)
        self.totals['scene_tag_pairs_flagged'] += len(flagged_labels)
        self.totals['scenes_flagged'] += bool(flagged_labels)

    def payload(self):
        total_keys = ('selected_scenes', 'analyzed_scenes', 'scenes_flagged',
                      'scene_tag_pairs_reviewed', 'scene_tag_pairs_flagged')
        return dict(
            totals={**{key: self.totals[key] for key in total_keys},
                    'unique_tags_reviewed': len(self.all_tags),
                    'unique_tags_flagged': len(self.review_tags),
                    'skipped_scenes': sum(self.skip_counts.values())},
            models=[dict(model=model, unique_tags_reviewed=len(self.inspected[model]),
                         unique_tags_flagged=len(self.flagged[model]),
                         scene_tag_pairs_reviewed=self.model_counts[model]['scene_tag_pairs_reviewed'],
                         scene_tag_pairs_flagged=self.model_counts[model]['scene_tag_pairs_flagged'],
                         scenes_flagged=self.model_counts[model]['scenes_flagged'],
                         reasons={reason: self.model_reasons[model][reason] for reason in self.reasons})
                    for model in SOURCES],
            tags=sorted(self.tags.values(), key=lambda entry: (
                -entry['scene_count'], -sum(entry['reasons'].values()), entry['model'], entry['tag'])),
            skipped_by_reason=dict(sorted(self.skip_counts.items())),
            skipped_scene_examples=self.skipped,
        )
