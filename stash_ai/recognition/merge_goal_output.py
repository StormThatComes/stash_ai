"""Reviewable replay provenance, explanations and relocatable configuration export."""
import hashlib
import json
from pathlib import Path
from .config import digest, plain
from .merge_report_policy import patched, proposal
from .merge_goals import unmet_conditions


def explain(result):
    simulation = result.get('simulation', {})
    chosen = result.get('selected_trial', 'baseline' if 'baseline' in simulation.get('trials', {}) else None)
    trial = simulation.get('trials', {}).get(chosen, {})
    status = trial.get('goal_result', {})
    result['explanation'] = dict(selected_trial=chosen, unresolved=unmet_conditions(status),
        conflicting_trials=[dict(trial=name, conflicts=item['goal_result']['conflicts'])
            for name, item in simulation.get('trials', {}).items() if not item['goal_result']['disjoint']],
        conclusion='Best among completed trials, including baseline; neither global optimality nor impossibility is proved.')
    feedback = trial.get('solver_feedback', {})
    if feedback:
        result['explanation']['solver_feedback'] = feedback
    result['explanation']['search_stop_reason'] = result.get('search', {}).get('stop_reason')
    if 'feasibility' in result:
        result['explanation']['feasibility_status'] = result['feasibility']['status']
    if simulation.get('input_selection', {}).get('excluded_scans'):
        result['explanation']['input_blockers'] = simulation['input_selection']['excluded_scans']
        result['explanation']['validation_scope'] = 'compatible_complete_latest_scenes_only'
        if simulation.get('stop_reason') == 'no_compatible_scans':
            result['explanation']['conclusion'] = 'No compatible complete latest scan is available; no trial was run.'
    hybrid = trial.get('hybrid_solver_evidence', {})
    if hybrid.get('enabled'):
        result['explanation']['hybrid'] = dict(
            coverage=trial.get('hybrid'), missing_identities=hybrid['missing_identities'],
            missing_evidence=[i for i in hybrid['identities'] if not i['person_references'] or
                i['missing_person_observations'] or i['missing_original_core_observations']],
            pair_blockers=[dict(left=p['left'], right=p['right'], blockers=p['blockers'],
                support=p['support'], face=p['face'], person=p['person']) for p in hybrid['pairs']
                if p['relation']=='within_set' and p['blockers']],
            nearest_competitors=hybrid['nearest_competitors'],
            interpretation='Frozen diagnostics for the selected replay; missing features require collection, '
                           'and guards/competitors are still checked by production matching.')


def configuration_document(config, recommendation, inputs, *, mode='full'):
    import yaml
    if mode not in ('full', 'diff'):
        raise ValueError('YAML mode must be full or diff.')
    changes = {x['parameter']:x['proposed'] for x in recommendation['changes']}
    settings = patched(config, changes)  # Validate changes against the effective input baseline.
    if mode == 'diff':
        data = proposal(config, changes)['yaml_patch']
    else:
        data = plain(settings.data)
        data['storage']['database'] = str(config.database)
        data['models']['directory'] = str(config.model_dir)
        data['names']['file'] = str(config.names_path) if config.names_path else None
        for section,key in (('model','source_directory'),('model','checkpoint_directory'),('pose','path')):
            raw=data['sapiensid'][section][key]
            if raw:
                path=Path(raw).expanduser()
                data['sapiensid'][section][key]=str((config.source_path.parent/path).resolve())
    serialized = yaml.safe_dump(data, sort_keys=True)
    if mode == 'diff':
        serialized = ('# YAML diff: merge these keys into the input baseline configuration.\n'
                      '# This is not a standalone --face-config document.\n'
                      f'# Baseline configuration SHA256: {digest(plain(config.data))}\n' + serialized)
    if 'quality' in recommendation:
        header = f"# Merge solver: best completed trial {recommendation['trial']} ({recommendation['quality']}).\n"
        diagnostics = recommendation.get('goal_diagnostics', {})
        if diagnostics:
            header += f"# Goal policy: {diagnostics['policy']['name']} v{diagnostics['policy']['version']}.\n"
            if diagnostics['policy']['name'] == 'identity':
                header += (f"# Strict all-sample success: {diagnostics['strict_fully_solved']}; "
                           f"strict disjoint: {diagnostics['strict_disjoint']}; "
                           f"soft units lost/reassigned: {diagnostics['soft_lost_units']}. See JSON evidence details.\n")
                header += ('# Preservation uses set scene coverage; member representation and soft budgets are diagnostics.\n')
            if diagnostics.get('scene_exclusions'):
                header += '# Explicit performer/scene exclusions were evaluated; see JSON. These are not runtime overrides.\n'
        if not recommendation['fully_solved']:
            reasons = sorted({r['reason'] for r in recommendation['unmet_conditions']})
            header += '# Unmet conditions: ' + ', '.join(reasons) + '. See the JSON report for details.\n'
        if inputs.get('excluded_scans'):
            header += (f"# Partial input scope: verified on {inputs['selected_scene_count']}/{inputs['latest_scene_count']} scenes.\n"
                       + ('# See simulation.selection in the JSON report; excluded or recollected scenes require revalidation.\n'
                          if mode == 'diff' else
                          '# Use run_argv from the JSON report; excluded or recollected scenes require revalidation.\n'))
        serialized = header + serialized
    # Validate the serialized full document before exposing it as a runnable config.
    from .config import parse_yaml, validate
    document = parse_yaml(serialized.encode())
    if mode == 'full':
        validate(document)
    return serialized


def export(config, result, output, yaml_output, *, yaml_owner=None, yaml_mode='full'):
    if yaml_mode not in ('full', 'diff'):
        raise ValueError('YAML mode must be full or diff.')
    targets = [p.expanduser().resolve() for p in (output, yaml_output) if p]
    protected = {config.database.resolve(), config.source_path}
    if len(set(targets)) != len(targets) or any((p.exists() and (yaml_owner is None or p != yaml_owner.path)) or p in protected for p in targets):
        raise FileExistsError('Outputs must be distinct new files, separate from source configuration and database.')
    recommendation = result.get('best_result') or result.get('recommendation')
    result['yaml_exported'] = False
    if yaml_output and recommendation:
        serialized = configuration_document(config, recommendation, result['simulation'].get('input_selection', {}),
                                            mode=yaml_mode)
        destination = yaml_output.expanduser().resolve()
        if yaml_owner is not None:
            yaml_owner.write(serialized)
        else:
            with destination.open('x', encoding='utf-8') as stream:
                stream.write(serialized)
        result['yaml_exported'] = True
        result['yaml_export_quality'] = recommendation.get('quality')
        result['yaml_export_mode'] = yaml_mode
        if yaml_mode == 'diff':
            result.pop('run_argv', None)
            result['yaml_diff_baseline'] = dict(configuration_path=str(config.source_path),
                configuration_sha256=digest(plain(config.data)),
                application='Merge changed leaf values into the input baseline, preserving other settings and input overrides.')
        else:
            result.pop('yaml_diff_baseline', None)
            result['run_argv'] = ['python','stash_ai.py','recognition','run','--face-config',str(destination)]
            for scene in result['simulation']['selection']:
                result['run_argv'].extend(['--scene-id', scene])
    code_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        code_hash.update(path.name.encode())
        code_hash.update(path.read_bytes())
    result['provenance'] = dict(code_sha256=code_hash.hexdigest(), snapshot_sha256=result.get('snapshot_sha256'), configuration_sha256=digest(plain(config.data)),
        source_sha256=hashlib.sha256(config.source).hexdigest(), namespace=result['namespace'],
        goals_sha256=digest(dict(groups=result['groups'], goal_policy=result.get('goal_policy'),
            **({'scene_exclusions': result['scene_exclusions']} if result.get('scene_exclusions') else {}))),
        validity='Same retained evidence, scope, model, code and joint scene selection; changed media requires revalidation.')
    explain(result)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if output:
        with output.expanduser().open('x', encoding='utf-8') as stream:
            stream.write(text)
    else:
        print(text, end='')


def group_summary(group):
    """One set's actual trial outcome; completeness and partial progress are distinct."""
    assigned = group['assigned'] / group['observations'] if group['observations'] else 0.
    members = group['members']
    components = json.dumps(group['complete_components'], separators=(',', ':'))
    identity = ''
    if 'required_scenes' in group:
        identity = (f"scenes_preserved={group['covered_scenes']}/{group['required_scenes']} "
                    f"set_shared_coverage={group['set_shared_coverage']:.1%} "
                    f"set_fully_merged={group['identity_fully_merged']} ")
    if 'consolidated_performers' in group:
        identity += (f"consolidated={group['consolidated_performers']}/{group['total_performers']} "
                    f"identity_components={json.dumps(group['identity_components'], separators=(',', ':'))} "
                    f"represented={group['represented_performers']}/{group['total_performers']} "
                    f"core_shared_coverage={group['core_shared_coverage']:.1%} "
                    f"soft_lost_or_reassigned={group['soft_lost_units']} soft_budget_excess={group['soft_budget_excess']} ")
    return (f"  Set {group['index']+1}: {identity}largest_complete={group['largest_complete_performers']}/{group['total_performers']} "
            f"complete_components={components} assigned={group['assigned']}/{group['observations']} ({assigned:.1%}) "
            f"shared_coverage={group['consolidation_fraction']:.1%} fragments={group['fragments']} "
            f"missing_obs={group['missing']} withheld_obs={group['withheld']} "
            f"missing_performers={sum(m['missing'] > 0 for m in members)} "
            f"withheld_performers={sum(m['withheld'] > 0 for m in members)} "
            f"split_performers={sum(len(m['resulting_identities']) > 1 for m in members)} "
            f"empty_performers={sum(m['observations'] == 0 for m in members)}")


def group_summary_lines(group):
    from .merge_goal_samples import lines
    return [group_summary(group)] + lines(group)


def summary(result):
    """Final stderr summary; JSON stdout and report files remain machine-readable."""
    selected = result.get('best_result')
    inputs = result.get('simulation', {}).get('input_selection', {})
    input_lines = []
    if inputs.get('excluded_scans'):
        from .replay_inputs import describe
        input_lines = ['Merge solver input scope is partial. ' + describe(inputs)]
        if selected is not None:
            scope_hint = 'simulation.selection' if result.get('yaml_export_mode') == 'diff' else 'run_argv'
            input_lines.append(f'Recommendations apply only to replayed scenes; use {scope_hint} from the JSON report.')
    if selected is None:
        from .merge_solver_feasibility import lines as feasibility_lines
        return '\n'.join(['Merge solver: no completed trial; no configuration selected.'] + input_lines +
                         (feasibility_lines(result['feasibility']) if 'feasibility' in result else []))
    name = selected['trial']
    goal = result['simulation']['trials'][name]['goal_result']
    lines = [f"Merge solver best completed trial: {name} ({selected['quality']}); "
             f"configuration={'changed' if selected['changed'] else 'unchanged'}; disjoint={goal['disjoint']}."]
    if goal.get('goal_policy', {}).get('name') == 'identity':
        lines.append(f"  Goal policy: identity; strict_disjoint={goal['strict_disjoint']}; "
                     f"strict_fully_solved={goal['strict_fully_solved']}; largest_complete counts all original samples.")
    lines.extend(line for group in goal['groups'] for line in group_summary_lines(group))
    from .merge_scene_exclusions import summary_lines
    lines.extend(summary_lines(goal))
    if 'feasibility' in result:
        from .merge_solver_feasibility import lines as feasibility_lines
        lines.extend(feasibility_lines(result['feasibility']))
    for violation in selected['unmet_conditions']:
        detail = []
        for key, value in violation.items():
            if key == 'reason':
                continue
            if key == 'set':
                value += 1
            if key == 'sets':
                value = [v+1 for v in value]
            detail.append(f'{key}={value}')
        lines.append('  Unmet: ' + violation['reason'] + ' (' + ', '.join(detail) + ').')
    lines.append('Best among tested configurations; global optimality is not proved.')
    if result.get('search', {}).get('stop_reason'):
        lines.append(f"Search stopped: {result['search']['stop_reason']}; completed "
                     f"{result['search']['executed']}/{result['search']['max_trials']} trials.")
    queries = result['simulation']['trials'][name].get('solver_feedback', {}).get('query', {})
    if queries.get('truncated'):
        lines.append('Query feedback is partial: ' + ', '.join(queries['truncation_reasons']) +
                     '; independent probes remain available.')
    lines.extend(input_lines)
    if result.get('yaml_export_mode') == 'diff':
        lines.append('YAML diff saved: merge changed keys into the input baseline before running recognition.')
    return '\n'.join(lines)
