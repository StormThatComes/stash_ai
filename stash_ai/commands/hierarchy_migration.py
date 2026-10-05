"""One-time local YAML v1 migration: python -m stash_ai.commands.hierarchy_migration FILE ..."""
from pathlib import Path
import argparse
import os
import tempfile

from stash_ai.config import hierarchy_from_data, load_unique_yaml
from stash_ai.models import SOURCES


def convert(data):
    if not isinstance(data, dict) or not isinstance(data.get('groups'), dict):
        raise ValueError('Invalid hierarchy document.')
    groups = data['groups']
    if any(not isinstance(rows, list) or any(not isinstance(label, str) for label in rows)
           for rows in groups.values()):
        raise ValueError('Invalid hierarchy label lists.')
    labels = sorted({label for rows in groups.values() for label in rows})
    if type(data.get('version')) is not int or data['version'] not in (1, 2):
        raise ValueError('Unsupported hierarchy version.')
    if data['version'] == 1:
        if set(data) - {'version', 'root', 'category_prefix', 'source', 'groups'}:
            raise ValueError('Unknown historical hierarchy fields.')
        root, prefix = data.get('root'), data.get('category_prefix')
        if any(not isinstance(value, str) or not value or value != value.strip()
               or '\n' in value or '\r' in value for value in (root, prefix)):
            raise ValueError('Invalid historical hierarchy root or prefix.')
        source = next((source for source in SOURCES.values() if prefix.startswith(source.prefix)), None)
        if source is None:
            raise ValueError('Unknown historical hierarchy namespace.')
        result = {key: value for key, value in data.items() if key not in ('root', 'category_prefix')}
        result['version'] = 2
    else:
        source = next(iter(SOURCES.values()))
        result = data
    hierarchy_from_data(result, labels, source)
    return result


def migrate(paths, *, dry_run=False):
    """Validate every input first; replace each file atomically and keep its v1 copy."""
    import yaml
    paths = [Path(path).expanduser().resolve() for path in paths]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate hierarchy paths.')
    updates = []
    for path in paths:
        original = path.read_bytes()
        data = load_unique_yaml(path, 'Hierarchy')
        converted = convert(data)
        if data['version'] == 2:
            continue
        backup = path.with_name(path.stem + '.v1' + path.suffix)
        if backup.exists() and backup.read_bytes() != original:
            raise ValueError(f'Existing migration backup differs: {backup}')
        content = yaml.safe_dump(converted, allow_unicode=True, sort_keys=False).encode('utf-8')
        updates.append((path, backup, original, content))
    if dry_run:
        return [path for path, *_ in updates]
    for path, backup, original, content in updates:
        if path.read_bytes() != original:
            raise RuntimeError(f'Hierarchy changed during migration: {path}')
        if not backup.exists():
            with backup.open('xb') as saved:
                saved.write(original)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as staged:
            staged.write(content)
            temporary = Path(staged.name)
        try:
            os.chmod(temporary, path.stat().st_mode & 0o777)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return [path for path, *_ in updates]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Migrate hierarchy YAML v1 to canonical AI categories (v2).')
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    for path in migrate(args.files, dry_run=args.dry_run):
        print(f'{"Would migrate" if args.dry_run else "Migrated"}: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
