"""ONNX microbatches only when both model tensor contracts are dynamic."""


def dynamic(session):
    inputs, outputs = session.get_inputs(), session.get_outputs()
    return (len(inputs) == len(outputs) == 1 and
            all(value.shape and not isinstance(value.shape[0], int) for value in (inputs[0], outputs[0])))


def infer(owner,tensors):
    from .inference_cache import many,current
    if current() is None:return _infer(owner,tensors)
    from .models import ASSETS
    if not hasattr(owner,'config'):return _infer(owner,tensors)
    model=owner.config.data['models']['recognizer']
    contract=['face-embedding-v1',model,ASSETS[model].sha256,getattr(owner,'provenance',{}),str(owner.device)]
    return many('face_embedding',contract,tensors,lambda values:_infer(owner,values))


def _infer(owner, tensors):
    import numpy as np
    from .models import normalized_embedding
    if not tensors:
        return []
    if any(t.shape != (1,3,112,112) or t.dtype != np.float32 for t in tensors):
        raise ValueError('Invalid prepared face input.')
    if len(tensors) > 1 and not dynamic(owner.session):
        return [_infer(owner, [tensor])[0] for tensor in tensors]
    tensor = tensors[0] if len(tensors) == 1 else np.concatenate(tensors, axis=0)
    output = owner.session.run([owner.output_name], {owner.input_name: tensor})
    if len(output) != 1 or output[0].shape != (len(tensors), owner.dimensions):
        raise ValueError('Face recognizer returned an incompatible output batch.')
    return [normalized_embedding(row, owner.dimensions) for row in output[0]]
