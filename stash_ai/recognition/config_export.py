"""Export a portable recognition policy without opening the database or models."""
from pathlib import Path

import yaml

from stash_ai.reports.io import report_stream

from .config import plain
from .config_layout import get, put, v2_document

LOCAL_PATHS = (
    'sapiensid.model.source_directory', 'sapiensid.model.checkpoint_directory',
    'sapiensid.pose.path', 'sapiensid.pose.vitpose_directory',
    'sapiensid.pose.sapiens_directory', 'sapiensid.pose.dfine_directory',
    'sapiensid.pose.pmpose_directory', 'sapiensid.pose.rfdetr_directory',
    'sapiensid.pose.refinement.directory',
)


def export_data(config, *, absolute_paths=False):
    data = plain(config.data)
    if absolute_paths:
        data['storage']['database'] = str(config.database)
        data['models']['directory'] = str(config.model_dir)
        data['names']['file'] = str(config.names_path) if config.names_path else None
        root = config.source_path.parent if config.source_path else Path.cwd()
        for key in LOCAL_PATHS:
            value = get(data, key)
            if value:
                put(data, key, str((root / Path(value).expanduser()).resolve()))
    return data


def migrate(config, destination):
    destination = destination.expanduser().resolve()
    if destination in (config.source_path, config.database, config.names_path):
        raise ValueError('Choose a new YAML output separate from the input, database and names file.')
    root = config.source_path.parent if config.source_path else Path.cwd()
    data = export_data(config, absolute_paths=destination.parent != root)
    document = v2_document(data)
    with report_stream(destination, sync_directory=True) as stream:
        yaml.safe_dump(document, stream, sort_keys=False)
    print(f'Wrote recognition configuration v2 to {destination}')
    return 0
