"""Bounded CPU/media lookahead; inference and SQLite stay on the caller."""
from .profiling import pipeline_stage
from collections import OrderedDict
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

    @pipeline_stage('frame_input_wait')
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
    """Ordered bounded lookahead; prepared inputs share the run sprite budget."""
    def __init__(self, client, config, source_factory, *, report=None):
        self.client, self.config, self.source_factory, self.report = client, config, source_factory, report
        from .sprite_cache import current
        self.sprite_cache = current()
        self.enabled = config.data['execution']['prefetch_scenes']
        self.executor = self.active = None
        self.queued = OrderedDict()
        self.depth = config.data['execution']['prefetch_scene_depth']

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
        from .sprite_cache import use as use_sprites
        with use(token), use_sprites(self.sprite_cache):
            return self._prepare(identifier, token)

    @property
    def identifier(self):
        return next(iter(self.queued), None)

    @property
    def pending(self):
        return self.queued[self.identifier][0] if self.queued else None

    @property
    def token(self):
        return self.queued[self.identifier][1] if self.queued else None

    def fill(self, identifiers):
        for identifier in identifiers:
            if len(self.queued) >= self.depth:
                break
            self.submit(identifier)

    def submit(self, identifier):
        if not self.enabled or identifier is None or identifier in self.queued or len(self.queued) >= self.depth:
            return
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=min(4,self.depth), thread_name_prefix='recognition-scenes')
        token = Cancellation()
        try:
            submit_owned(self.executor, lambda: self._background(identifier, token),
                         lambda future: self.queued.__setitem__(identifier, (future, token)))
        except BaseException:
            self.close()
            raise

    @pipeline_stage('scene_input_wait')
    def take(self, identifier):
        if not self.enabled:
            self.active = self._prepare(identifier)
            return self.active
        if self.pending is None or self.identifier != identifier:
            raise RuntimeError('Recognition scene prefetch order changed.')
        try:
            result = self.pending.result()
        except Exception:
            self.queued.pop(identifier)
            raise
        # KeyboardInterrupt keeps the future registered until cleanup owns it.
        self.active = result
        self.queued.pop(identifier)
        return result

    def _close_future(self, future):
        from stash_ai.runtime.shutdown import join, release, ShutdownIncomplete
        if future.cancelled():
            return
        try:
            join(future, self)
            result = future.result()
        except ShutdownIncomplete:
            release(future, lambda: self._close_future(future))
            raise
        except BaseException:
            return  # Unconsumed speculative failure cannot fail a completed scene.
        result.close()

    def _close_pending(self):
        cleanup([lambda future=future:self._close_future(future) for future,_ in self.queued.values()])

    def close(self):
        from stash_ai.runtime.shutdown import call, release
        def reset():
            self.executor = self.active = None
            self.queued.clear()
        cleanup([
            ('cancel recognition input tokens', lambda:cleanup([token.cancel for _,token in self.queued.values()])),
            ('cancel recognition scene tasks', lambda:cleanup([future.cancel for future,_ in self.queued.values()])),
            ('close recognition pending inputs', self._close_pending),
            ('stop recognition scene readers', lambda: call(lambda: self.executor.shutdown(wait=True, cancel_futures=True), self)
             if self.executor is not None else None),
            ('close recognition active input', lambda: self.active.close() if self.active is not None else None),
            ('release recognition prefetch state', lambda: release(self, reset)),
        ], report=self.report)
