"""Reuse verified YOLO box weights without executing or retaining pose heads."""


def detection_only(model):
    from ultralytics.nn.modules.head import Detect, Pose
    head = model.model.model[-1]
    if not isinstance(head, Pose):
        raise ValueError('Top-down pose requires a supported YOLO person detection head.')
    # Pose/Pose26 extend Detect; the box/class branches and decoding state are
    # identical. Switch to its base implementation before predictor creation.
    # No new checkpoint, random weights or private forward monkey-patch.
    head.__class__ = Detect
    for name in tuple(head._modules):
        if name.startswith(('cv4', 'one2one_cv4')) or name == 'flow_model':
            delattr(head, name)
    for name in ('kpt_shape', 'nk', 'nk_sigma'):
        if hasattr(head, name):delattr(head, name)
    model.task = model.model.task = 'detect'
    model.overrides['task'] = 'detect'
    model.predictor = None
    return model
