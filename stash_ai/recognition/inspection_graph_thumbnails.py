"""Read bounded graph crops on a worker; never construct Tk images or run models."""
from contextlib import closing
from PIL import Image
from .identity_inspection import _image
from .inspection_runtime import check_cancelled, open_readonly, stamp
from .inspection_graph import MAX_NODES

THUMBNAIL_SIZE = 64


def load(config, graph, progress=lambda done,total: None):
    """Decode each retained crop once and discard full-resolution pixels promptly.

    One consistent snapshot, no network reconstruction and no persistent writes.
    Missing/historical sources retain the scoped caption returned by the reader.
    """
    revision = stamp(config.database)
    if revision != graph['revision']:
        raise ValueError('Recognition changed. Refresh the graph before loading thumbnails.')
    nodes = [n for n in graph['nodes'] if n['type']=='observation']
    if len(nodes)>MAX_NODES:
        raise ValueError('Graph thumbnail budget exceeded.')
    images, captions = {}, {}
    progress(0,len(nodes))
    with closing(open_readonly(config.database)) as db:
        for done,node in enumerate(nodes,1):
            check_cancelled()
            if node.get('availability')=='unavailable_in_scope':
                image,caption = None,'Observation unavailable in this scope and model.'
            else:
                image,caption = _image(db,graph['person'],node['kind'],node['observation'])
            if image is not None:
                try:
                    image.thumbnail((THUMBNAIL_SIZE,THUMBNAIL_SIZE),Image.Resampling.LANCZOS)
                    images[node['id']] = image.copy()
                finally:
                    image.close()
            captions[node['id']] = caption
            progress(done,len(nodes))
    check_cancelled()
    if stamp(config.database)!=revision:
        raise ValueError('Recognition changed during thumbnail loading. Refresh the graph.')
    return dict(images=images,captions=captions,total=len(nodes),missing=len(nodes)-len(images))
