"""Reviewable replay provenance, explanations and relocatable configuration export."""
import hashlib
import json
from pathlib import Path
from .config import digest, plain
from .merge_report_policy import patched


def explain(result):
    chosen = result.get('selected_trial') or 'baseline'
    simulation = result.get('simulation', {})
    trial = simulation.get('trials', {}).get(chosen, {})
    status = trial.get('goal_result', {})
    reasons = []
    for group in status.get('groups', []):
        for member in group['members']:
            for condition, reason in ((member['missing'], 'missing_latest_observations'),
                (member['withheld'], 'withheld_observations'),
                (len(member['resulting_identities']) > 1, 'performer_split')):
                if condition:
                    reasons.append(dict(set=group['index'], identity=member['identity_id'], reason=reason))
        if not group['fully_merged']:
            reasons.append(dict(set=group['index'], reason='remaining_fragments', fragments=group['fragments']))
    result['explanation'] = dict(selected_trial=chosen, unresolved=reasons,
        conflicting_trials=[dict(trial=name, conflicts=item['goal_result']['conflicts'])
            for name, item in simulation.get('trials', {}).items() if not item['goal_result']['disjoint']],
        conclusion='Results describe tested configurations; search exhaustion is not proof of impossibility.')
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


def export(config, result, output, yaml_output):
    import yaml
    targets = [p.expanduser().resolve() for p in (output, yaml_output) if p]
    protected = {config.database.resolve(), config.source_path}
    if len(set(targets)) != len(targets) or any(p.exists() or p in protected for p in targets):
        raise FileExistsError('Outputs must be distinct new files, separate from source configuration and database.')
    recommendation = result.get('recommendation')
    result['yaml_exported'] = False
    if yaml_output and recommendation:
        settings = patched(config, {x['parameter']:x['proposed'] for x in recommendation['changes']})
        data = plain(settings.data)
        data['storage']['database'] = str(config.database)
        data['models']['directory'] = str(config.model_dir)
        data['names']['file'] = str(config.names_path) if config.names_path else None
        for section,key in (('model','source_directory'),('model','checkpoint_directory'),('pose','path')):
            raw=data['sapiensid'][section][key]
            if raw:
                path=Path(raw).expanduser()
                data['sapiensid'][section][key]=str((config.source_path.parent/path).resolve())
        serialized = yaml.safe_dump(data, sort_keys=False)
        # Validate the serialized full document before exposing it as a runnable config.
        from .config import parse_yaml, validate
        validate(parse_yaml(serialized.encode()))
        destination = yaml_output.expanduser().resolve()
        with destination.open('x', encoding='utf-8') as stream:
            stream.write(serialized)
        result['yaml_exported'] = True
        result['run_argv'] = ['python','stash_ai.py','recognition','run','--face-config',str(destination)]
        for scene in result['simulation']['selection']:
            result['run_argv'].extend(['--scene-id', scene])
    code_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        code_hash.update(path.name.encode())
        code_hash.update(path.read_bytes())
    result['provenance'] = dict(code_sha256=code_hash.hexdigest(), snapshot_sha256=result.get('snapshot_sha256'), configuration_sha256=digest(plain(config.data)),
        source_sha256=hashlib.sha256(config.source).hexdigest(), namespace=result['namespace'],
        goals_sha256=digest(result['groups']),
        validity='Same retained evidence, scope, model, code and joint scene selection; changed media requires revalidation.')
    explain(result)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if output:
        with output.expanduser().open('x', encoding='utf-8') as stream:
            stream.write(text)
    else:
        print(text, end='')
