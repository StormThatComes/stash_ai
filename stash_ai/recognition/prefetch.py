"""Bounded CPU/media lookahead; inference and SQLite stay on the caller."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from queue import Empty, Full
from threading import Thread

from stash_ai.inference.managed_stream import ManagedStream, StreamFailure
from stash_ai.runtime.cancellation import Cancellation, use
from stash_ai.runtime.cleanup import closing, cleanup
from stash_ai.runtime.startup import submit_owned


class FramePrefetch:
    def __init__(self, source, start, limit):
        self.source, self.start = source, start
        self.channel = ManagedStream(limit)

    def launch(self):
        try:
            self.channel.worker = Thread(target=self._produce, name='recognition-frames', daemon=True)
            self.channel.worker.start()
            return self
        except BaseException:
            self.close()
            raise

    def _send_image(self, value):
        while not self.channel.stopped.is_set():
            try:
                self.channel.queue.put(value, timeout=.05)
                return True
            except Full:
                pass
        return False

    def _produce(self):
        try:
            with use(self.channel.cancellation), closing(self.source.frames(self.start),
                                                        name='close recognition frame generator') as frames:
                for index, cue, image in frames:
                    self.channel.cancellation.check()
                    # SpriteInput's yielded crop expires when the generator
                    # advances. The consumer needs its own image across threads.
                    owned = image.copy()
                    transferred = False
                    try:
                        transferred = self._send_image((index, cue, owned))
                    finally:
                        if not transferred:
                            owned.close()
                    if not transferred:
                        break
        except BaseException as error:
            self.channel.send(StreamFailure(error))
        finally:
            self.channel.send(self.channel.end)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.channel.receive()
        if isinstance(value, StreamFailure):
            raise value.error
        if value is self.channel.end:
            raise StopIteration
        return value

    def _drain(self):
        images = []
        while True:
            try:
                value = self.channel.queue.get_nowait()
            except Empty:
                break
            if isinstance(value, tuple):
                images.append(value[2].close)
        cleanup(images)

    def close(self):
        from stash_ai.runtime.shutdown import release
        cleanup([self.channel.stop, self.channel.join, lambda: release(self, self._drain)])


def frames(source, start, limit):
    if not limit:
        return source.frames(start)
    return FramePrefetch(source, start, limit).launch()


class _SceneInput:
    def __init__(self, scene, source, stack):
        self.scene, self.source, self.stack = scene, source, stack

    def __enter__(self):
        return self

    def __exit__(self, *args):
        from stash_ai.runtime.shutdown import release
        release(self, lambda: self.stack.__exit__(*args))

    def close(self):
        from stash_ai.runtime.shutdown import release
        release(self, self.stack.close)


class ScenePrefetch:
    """Retain the active input plus at most one verified next-scene input."""
    def __init__(self, client, config, source_factory, *, report=None):
        self.client, self.config, self.source_factory, self.report = client, config, source_factory, report
        self.enabled = config.data['execution']['prefetch_scenes']
        self.executor = self.pending = self.active = self.token = self.identifier = None

    def _prepare(self, identifier, token=None):
        stack = ExitStack()
        try:
            reader = self.client
            if token is not None:
                reader = stack.enter_context(closing(self.client.new_media_reader(), report=self.report,
                                                    name='close recognition input HTTP'))
            scene = reader.scene(identifier)
            source = stack.enter_context(self.source_factory(reader, scene, self.config))
            if token is not None:
                token.check()
            return _SceneInput(scene, source, stack)
        except BaseException:
            cleanup([('close failed recognition input', stack.close)], report=self.report)
            raise

    def _background(self, identifier, token):
        with use(token):
            return self._prepare(identifier, token)

    def submit(self, identifier):
        if not self.enabled or identifier is None or self.pending is not None:
            return
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='recognition-scenes')
        self.identifier, self.token = identifier, Cancellation()
        token = self.token
        try:
            submit_owned(self.executor, lambda: self._background(identifier, token),
                         lambda future: setattr(self, 'pending', future))
        except BaseException:
            self.close()
            raise

    def take(self, identifier):
        if not self.enabled:
            self.active = self._prepare(identifier)
            return self.active
        if self.pending is None or self.identifier != identifier:
            raise RuntimeError('Recognition scene prefetch order changed.')
        try:
            result = self.pending.result()
        except Exception:
            self.pending = self.identifier = self.token = None
            raise
        # Register ownership before dropping the future, also if Ctrl+C lands
        # between returning the prepared input and entering its context.
        self.active = result
        self.pending = self.identifier = self.token = None
        return result

    def _close_pending(self):
        from stash_ai.runtime.shutdown import join, release, ShutdownIncomplete
        if self.pending is not None and not self.pending.cancelled():
            try:
                join(self.pending, self)
                result = self.pending.result()
            except ShutdownIncomplete:
                release(self.pending, self._close_pending)
                raise
            except BaseException:
                # Unconsumed speculative work cannot fail a completed scene.
                return
            result.close()

    def close(self):
        from stash_ai.runtime.shutdown import call, release
        def reset():
            self.executor = self.pending = self.active = self.token = self.identifier = None
        cleanup([
            ('cancel recognition next-scene input', lambda: self.token.cancel() if self.token is not None else None),
            ('cancel recognition scene task', lambda: self.pending.cancel() if self.pending is not None else None),
            ('close recognition pending input', self._close_pending),
            ('stop recognition scene reader', lambda: call(lambda: self.executor.shutdown(wait=True, cancel_futures=True), self)
             if self.executor is not None else None),
            ('close recognition active input', lambda: self.active.close() if self.active is not None else None),
            ('release recognition prefetch state', lambda: release(self, reset)),
        ], report=self.report)
