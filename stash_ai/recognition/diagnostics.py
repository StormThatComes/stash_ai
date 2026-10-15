"""Bounded, sanitized error evidence shared by console and recognition history."""
import re

from stash_ai.stash_api.transport import safe_error
from .media import safe_url

GROUP_LIMIT = 6
EXAMPLE_LIMIT = 2


def message(error, api_key=''):
    text = safe_error(error, api_key)
    def redact(match):
        try:
            return safe_url(match.group())
        except ValueError:
            return '[invalid recognition URL]'
    return re.sub(r'(?:https?:)?//[^\s\"\'<>]+', redact, text, flags=re.IGNORECASE)


def metadata(value, api_key=''):
    """Sanitize nested cached exclusions/quality errors without mutating history."""
    def clean(item, depth=0):
        if depth >= 64:
            return '[metadata nesting limit]'
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                if key == 'source' and isinstance(child, str):
                    try:
                        child = safe_url(child)
                    except ValueError:
                        child = '[invalid recognition URL]'
                result[key] = clean(child, depth + 1)
            return result
        if isinstance(item, (list, tuple)):
            return [clean(child, depth + 1) for child in item]
        return message(item, api_key) if isinstance(item, str) else item
    return clean(value)


def snapshot(report, api_key=''):
    groups, overflow = [], report.overflow
    for (stage, source, kind), group in report.groups.items():
        if len(groups) >= GROUP_LIMIT:
            overflow += group['count']
            continue
        groups.append({'stage': message(stage, api_key)[:80],
                       'source': message(source, api_key)[:80],
                       'type': message(kind, api_key)[:80], 'count': group['count'],
                       'examples': [message(value, api_key).replace('\n', ' ')[:350]
                                    for value in group['examples'][:EXAMPLE_LIMIT]]})
    return {'format': 1, 'total': sum(group['count'] for group in groups) + overflow,
            'groups': groups, 'overflow': overflow}


def render(report, errors):
    for group in errors.get('groups', []):
        report.write(f"Recognition errors {group['source']}/{group['type']}: {group['count']}")
        for example in group['examples']:
            report.write('  ' + example)
    if errors.get('overflow'):
        report.write(f"Other recognition errors: {errors['overflow']} (group budget exceeded).")
