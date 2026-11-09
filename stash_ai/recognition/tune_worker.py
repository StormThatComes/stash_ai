"""Keep desktop I/O and inference off the Tk thread. Only the UI polls completions."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

from .detection_session import open_scene, read_cue


class Worker:
    """One owner thread, one running operation and at most one latest pending request."""
    def __init__(self, name):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self.future = self.pending = self.callbacks = None

    def request(self, operation, complete, failed):
        if self.future is None:
            self.future = self.executor.submit(operation)
            self.callbacks = complete, failed
        else:
            self.callbacks = None
            self.pending = operation, complete, failed

    def invalidate(self):
        self.pending = self.callbacks = None

    def poll(self):
        if self.future is None or not self.future.done():
            return
        future, callbacks = self.future, self.callbacks
        pending = self.pending
        self.future = self.pending = self.callbacks = None
        if pending:
            self.request(*pending)
        if callbacks:
            complete, failed = callbacks
            try:
                result = future.result()
            except Exception as error:
                failed(error)
            else:
                complete(result)

    def close(self, release):
        self.invalidate()
        try:
            self.executor.submit(release).result()
        finally:
            self.executor.shutdown(wait=True)
            self.future = None


class SceneBrowser:
    def __init__(self, services, config):
        self.services = services
        self.config = config
        self.stack = ExitStack()
        self.selection = self.source = None
        self.times = ()

    def preview(self, selection, scene_id, index):
        if selection != self.selection:
            self.close()
            source = self.stack.enter_context(open_scene(self.services, self.config, scene_id))
            self.times = tuple((cue.start, cue.end) for cue in source.cues)
            self.selection, self.source = selection, source
        return self.times, read_cue(self.source, scene_id, index)

    def close(self):
        self.selection = self.source = None
        self.times = ()
        self.stack.close()
