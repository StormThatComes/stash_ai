"""Coverage counters and issues from assessed scene/model pairs."""
from collections import Counter, defaultdict

from stash_ai.reports.cache_assessment import REUSABLE, STATUSES
from stash_ai.reports.cache_plan import backfill_action
from stash_ai.state_results import KINDS

PAIR_COUNTS = (
    'evidence_ready_pairs', 'reusable_pairs', 'inference_pairs',
    'rethreshold_pairs', 'blocked_pairs',
)


class CoverageAccumulator:
    def __init__(self, models):
        self.models = models
        self.counts = {model: {kind: Counter() for kind in KINDS} for model in models}
        self.pairs = {model: Counter() for model in models}
        self.issues = []
        self.operations = defaultdict(list)
        self.blocked_scenes = set()
        self.complete_scenes = 0

    def record(self, assessments):
        scene_ready = True
        for pair in assessments:
            model = pair.model
            for kind, output in pair.outputs.items():
                self.counts[model][kind][output['status']] += 1
            statuses = {value['status'] for value in pair.outputs.values()}
            reusable = statuses <= REUSABLE
            ready = statuses == {'ready'}
            counts = self.pairs[model]
            counts['evidence_ready_pairs'] += ready
            counts['reusable_pairs'] += reusable
            counts['blocked_pairs'] += bool(pair.blockers)
            scene_ready &= ready and not pair.blockers
            if ready and not pair.blockers:
                continue
            row = dict(scene_id=pair.scene_id, model=model,
                       outputs=pair.outputs, blockers=pair.blockers)
            if pair.publication_dependencies:
                row['publication_dependencies'] = pair.publication_dependencies
            if 'blocked_selected_model' in pair.blockers:
                row['blocking_models'] = pair.blocking_models
            if pair.checkpoint:
                row['checkpoint'] = pair.checkpoint
            self.issues.append(row)
            if pair.blockers:
                self.blocked_scenes.add(pair.scene_id)
                continue
            action = backfill_action(statuses, pair.checkpoint)
            counts['rethreshold_pairs' if action == 'rethreshold' else 'inference_pairs'] += 1
            self.operations[action, model].append(pair.scene_id)
        self.complete_scenes += scene_ready

    def payload(self, selected_scenes):
        summary = [dict(model=model, **{key: self.pairs[model][key] for key in PAIR_COUNTS},
                        outputs={kind: {status: self.counts[model][kind][status] for status in STATUSES}
                                 for kind in KINDS}) for model in self.models]
        return dict(
            totals=dict(scenes=selected_scenes, fully_ready_scenes=self.complete_scenes,
                        scene_model_pairs=selected_scenes * len(self.models),
                        **{key: sum(row[key] for row in summary) for key in PAIR_COUNTS}),
            models=summary, issues=self.issues,
        )
