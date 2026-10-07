"""Offline explanations of actual recorded scene/model reuse decisions."""
from collections import Counter

from stash_ai.history.store import HistoryStore, history_path
from stash_ai.reports.io import write_json_report
from stash_ai.storage.store import scope_for


def run(app, args):
    models = args.models or ([args.run_model] if args.run_model else None)
    if args.models and args.run_model:
        models = [model for model in models if model == args.run_model]
        if not models:
            raise ValueError('--models and --run-model select no common model.')
    with HistoryStore(history_path(args), readonly=True) as store:
        decisions = store.decisions(scope_for(app.STASH_URL), limit=args.run_limit,
            scenes=args.scene_id, models=models, command=args.run_command, status=args.run_status)
    summaries = {}
    for item in decisions:
        row = summaries.setdefault(item['model'], dict(model=item['model'], decisions=0, actions=Counter(),
            statuses=Counter(), reasons=Counter(), scenes=set()))
        row['decisions'] += 1
        row['actions'][item['action']] += 1
        row['statuses'][item['status']] += 1
        row['reasons'].update(item['reasons'])
        row['scenes'].add(item['scene'])
    for row in summaries.values():
        row['scenes'] = len(row['scenes'])
    report = dict(format=1, decisions=decisions, models=list(summaries.values()),
        notes=['Records explain actual past decisions, not whether current Stash/model assets still match.',
               'Only the most recent run-limit matching runs with recorded decisions are selected.',
               'Reason counts can overlap; one decision is stored per run/scene/model.',
               'historical_identity_unavailable means earlier runs did not capture a comparable proof.',
               'Diagnostic proofs never authorize reuse and are not needed to resume inference.'])
    destination = write_json_report(report, args.reuse_report)
    print(f'Reuse decisions: {len(decisions)}')
    for row in report['models']:
        print(f"  {row['model']}: scenes={row['scenes']}, decisions={row['decisions']}; "
              + ', '.join(f'{key}={value}' for key, value in sorted(row['actions'].items())))
        print('    Reasons: ' + ', '.join(f'{key}={value}' for key, value in sorted(row['reasons'].items())))
    print('Report: ' + str(destination))
    return 0
