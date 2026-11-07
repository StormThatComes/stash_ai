"""Detailed per-trial diagnostics for verbose console output."""


def verbose_progress(maximum):
    from .merge_goal_output import group_summary_lines
    import json
    import sys
    completed_trials = 0
    def progress(name, stage, item):
        nonlocal completed_trials
        detail = ''
        if stage == 'start':
            if item['previous_trial'] is None:
                detail = (' (input configuration with capped support)' if item['changes_from_previous']
                          else ' (input configuration)')
            else:
                detail = f" changes_vs={item['previous_trial']} parent={item['search_parent']}"
        elif stage == 'complete':
            completed_trials += 1
            detail = (f" disjoint={item['goal_result']['disjoint']} "
                      f"best={item['best_trial']} best_number={item['best_trial_number']}")
            if item['goal_result'].get('goal_policy', {}).get('name') == 'identity':
                detail += f" goal_policy=identity strict_disjoint={item['goal_result']['strict_disjoint']}"
            detail += f" elapsed={item['elapsed_seconds']:.1f}s completed={completed_trials}/{maximum}"
        elif stage in ('plan', 'pool', 'status', 'ready'):
            detail = ' ' + item['text']
        elif stage.startswith('stage: ') and item:
            from .solver_progress import worker_text
            import time
            detail = ' | ' + worker_text(item, time.monotonic())
        print(f'Merge solver {name}: {stage}{detail}', file=sys.stderr, flush=True)
        if stage == 'start':
            for change in item['changes_from_previous']:
                print(f"  {change['parameter']}: {json.dumps(change['previous'])} -> "
                      f"{json.dumps(change['current'])}", file=sys.stderr, flush=True)
            if item.get('search_phase') == 'bidirectional':
                print(f"  Search: phase=bidirectional strategy={item['search_strategy']}",
                      file=sys.stderr, flush=True)
            if item.get('search_phase') == 'feasibility':
                print(f"  Search: phase=feasibility profile={item['search_strategy']}", file=sys.stderr, flush=True)
            if item.get('search_phase') == 'anchor_rules':
                print(f"  Anchor rule profile: {item['search_strategy']}", file=sys.stderr, flush=True)
            if item.get('search_phase') == 'targeted':
                target = item['search_target']
                print(f"  Search: phase=targeted set={target['set']+1} route={target['route']} "
                      f"performers={json.dumps(target['performers'])} blockers={json.dumps(target['blockers'])}",
                      file=sys.stderr, flush=True)
        elif stage == 'complete':
            for group in item['goal_result']['groups']:
                print('\n'.join(group_summary_lines(group)), file=sys.stderr, flush=True)
            from .merge_scene_exclusions import summary_lines
            for line in summary_lines(item['goal_result']):
                print(line, file=sys.stderr, flush=True)
            construction = item['goal_result'].get('construction')
            if construction:
                merged = [g['index']+1 for g in construction['groups'] if g['merged']]
                print(f"  Construction: {construction['status']} merged_sets={merged} "
                      f"goals_satisfied={construction['goals_satisfied']}", file=sys.stderr, flush=True)
        elif stage == 'failed':
            print(f"  Trial incomplete: {item['reason']}; stage={item['stage']}; "
                  f"elapsed={item['elapsed_seconds']:.1f}s; {item['error']}", file=sys.stderr, flush=True)
    return progress
