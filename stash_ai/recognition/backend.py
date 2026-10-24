"""Invocation-owned identity models borrowed by scans and video portraits."""
from stash_ai.runtime.cleanup import cleanup
from contextlib import contextmanager, nullcontext


def prepared_frame(backend, image):
    factory = getattr(backend, 'prepared_frame', None)
    return factory(image) if factory is not None else nullcontext()


class LazyBackend:
    def __init__(self, config, paths, factory, *, loaded=lambda model: None):
        self.config, self.paths, self.factory, self.loaded = config, paths, factory, loaded
        self.model = None
        self.closed = False

    def borrow(self, config=None, paths=None):
        """Accept the scan/video factory signatures without transferring ownership."""
        from stash_ai.runtime.shutdown import require_idle
        require_idle()
        if self.closed:
            raise RuntimeError('Recognition backend owner is closed.')
        if self.model is None:
            model = self.factory(self.config, self.paths)
            self.model = model
            try:
                self.loaded(model)
            except BaseException:
                self.invalidate(model)
                raise
        return _BorrowedBackend(self, self.model)

    def invalidate(self, model):
        # A failed inference must not poison later scenes. Clear ownership even
        # when close fails; cleanup preserves the original inference exception.
        if self.model is model:
            self.model = None
            cleanup([('close failed recognition backend', model.close)])

    def close(self):
        from stash_ai.runtime.shutdown import release
        release(self, self._close)

    def _close(self):
        self.closed = True
        if self.model is not None:
            self.invalidate(self.model)


class _BorrowedBackend:
    def __init__(self, owner, model):
        self.owner, self.model = owner, model

    @property
    def active(self):
        return self.model is not None and self.owner.model is self.model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def _infer(self, method, *args):
        from stash_ai.runtime.shutdown import require_idle
        require_idle()
        if self.model is None or self.owner.model is not self.model:
            raise RuntimeError('Recognition backend lease is no longer active.')
        try:
            return getattr(self.model, method)(*args)
        except BaseException:
            self.owner.invalidate(self.model)
            raise

    def detect(self, image):
        return self._infer('detect', image)

    def embed(self, image, face):
        return self._infer('embed', image, face)

    def embed_prepared_many(self, tensors):
        return self._infer('embed_prepared_many', tensors)

    @contextmanager
    def prepared_frame(self, image):
        if not self.active:
            raise RuntimeError('Recognition backend lease is no longer active.')
        if not hasattr(self.model, 'prepare_frame'):
            yield
            return
        previous = self._infer('prepare_frame', image)
        try:
            yield
        finally:
            if self.active:
                cleanup([('release prepared recognition frame', lambda: self._infer('release_frame', previous))])

    def close(self):
        # Scene/portrait cleanup releases only its lease. The run owns weights.
        self.model = None
