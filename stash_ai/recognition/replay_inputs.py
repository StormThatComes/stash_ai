"""Explain retained-input availability without weakening scan contracts."""


class ReplayInputError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('Replay input validation failed before baseline.\n' + describe(diagnostics))


def describe(diagnostics):
    excluded = diagnostics['excluded_scans']
    if not excluded:
        return 'All latest retained scans are compatible and complete.'
    lines = [f"Compatible complete scenes: {diagnostics['selected_scene_count']}/{diagnostics['latest_scene_count']}."]
    if 'configuration_path' in diagnostics:
        lines.append('Input configuration: ' + (diagnostics['configuration_path'] or 'built-in defaults (no YAML file)') + '.')
    if 'database_path' in diagnostics:
        lines.append('Input database: ' + diagnostics['database_path'] + '.')
    for reason in ('scan_policy_mismatch', 'incomplete_scan'):
        scenes = [s['scene'] for s in excluded if reason in s['reasons']]
        if scenes:
            lines.append(reason + ': scenes=' + ','.join(scenes) + '.')
    if any('scan_policy_mismatch' in s['reasons'] for s in excluded):
        lines.append('Scan policy covers detection settings, execution.max_cues and visibility measurements; '
                     'sapiensid.matching.* changes alone do not require a rescan. '
                     'Use the original scan settings or run recognition for the listed scenes with the current configuration.')
        lines.append('Pass the same --face-config to run and report-merge; --yaml-output is only an output path, '
                     'not the input configuration. A run that reuses an older matching scan makes it current again.')
    if any('incomplete_scan' in s['reasons'] for s in excluded):
        lines.append('Run recognition for the incomplete scenes to finish collecting their observations.')
    return '\n'.join(lines)


def classify(scans, expected_policy):
    selected, excluded = [], []
    for scan in scans:
        reasons = []
        if scan['policy_sha'] != expected_policy:
            reasons.append('scan_policy_mismatch')
        if (scan['status'] != 'complete' or type(scan['cursor']) is not int or type(scan['total']) is not int or
                not 0 <= scan['cursor'] == scan['total']):
            reasons.append('incomplete_scan')
        if reasons:
            excluded.append(dict(scene=scan['scene'], scan=scan['id'], reasons=reasons,
                status=scan['status'], cursor=scan['cursor'], total=scan['total'],
                stored_policy_sha256=scan['policy_sha'], expected_policy_sha256=expected_policy))
        else:
            selected.append(scan)
    return selected, dict(latest_scene_count=len(scans), selected_scene_count=len(selected),
        excluded_scene_count=len(excluded), complete_scope=not excluded, excluded_scans=excluded,
        interpretation='Only compatible complete latest scans are replayed. Excluded scenes stay in frozen history '
                       'and remain subject to production reference admission. Latest means the current scan verified '
                       'by run, or the newest created scan when none was explicitly reselected; replay never falls back to older evidence.')
