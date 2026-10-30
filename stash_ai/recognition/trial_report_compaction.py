"""Lossless body identity deltas for intermediate solver trials."""
ENCODING = 'baseline-identities-delta-v1'


def key(row):
    return row['person'], row['status']


def compact(report, baseline):
    if 'identities' not in report:
        return report
    before = {key(row): row for row in baseline}
    after = {key(row): row for row in report['identities']}
    delta = dict(removed=[list(k) for k in before if k not in after],
                 changed=[row for k, row in after.items() if before.get(k) != row])
    # A dramatically changed partition can be smaller as a complete list.
    from .database import encoded
    if len(encoded(delta)) >= len(encoded(report['identities'])):
        return report
    result = dict(report)
    result.pop('identities')
    result.update(identities_encoding=ENCODING, identities_delta=delta,
                  identities_count=len(after))
    return result


def expand(report, baseline):
    if report.get('identities_encoding') != ENCODING:
        return report
    values = {key(row): row for row in baseline}
    delta = report['identities_delta']
    for address in delta['removed']:
        values.pop(tuple(address))
    values.update((key(row), row) for row in delta['changed'])
    if len(values) != report['identities_count']:
        raise ValueError('Invalid body identity delta.')
    result = {k:v for k,v in report.items() if k not in ('identities_encoding','identities_delta','identities_count')}
    result['identities'] = sorted(values.values(), key=lambda r:(r['person'] is not None,r['person'] or '',r['status']))
    return result
