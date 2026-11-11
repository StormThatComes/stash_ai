"""One admission budget for retained sprites, verified bytes and CPU weights."""
from contextlib import contextmanager
from contextvars import ContextVar
from collections import OrderedDict
from threading import RLock

from .profiling import pipeline_count, pipeline_max, pipeline_timing

_active = ContextVar('recognition_retained_memory', default=None)


class Budget:
    def __init__(self, limit, model_limit=0):
        self.limit, self.model_limit = limit, model_limit
        self.lock = RLock()
        self.owners, self.models = {}, OrderedDict()
        self.used = self.model_bytes = 0

    def replace(self, owner, amount):
        with self.lock:
            used = self.used - self.owners.get(owner, 0) + amount
            if used > self.limit:
                pipeline_count('retained_memory_admission_misses')
                return False
            if amount:
                self.owners[owner] = amount
            else:
                self.owners.pop(owner, None)
            self.used = used
            pipeline_max('retained_memory_peak_bytes', used)
            return True

    def verified(self, key, load):
        with self.lock:
            blob = self.models.pop(key, None)
            if blob is not None:
                self.models[key] = blob
                pipeline_count('verified_model_bytes_hits')
                return blob
            with pipeline_timing('model_asset_read_verify'):
                blob = load()
            pipeline_count('verified_model_bytes_misses')
            size = len(blob)
            if size > self.model_limit:
                return blob
            while self.models and (self.model_bytes + size > self.model_limit or self.used + size > self.limit):
                old, value = self.models.popitem(last=False)
                self.model_bytes -= len(value)
                self.replace(('verified', old), 0)
            if self.replace(('verified', key), size):
                self.models[key] = blob
                self.model_bytes += size
            return blob

    def close(self):
        with self.lock:
            self.models.clear()
            self.owners.clear()
            self.used = self.model_bytes = 0


def current():
    return _active.get()


@contextmanager
def scope(config):
    settings = config.data['execution']
    budget = Budget(settings['retained_memory_bytes'], settings['model_bytes_cache_bytes'])
    token = _active.set(budget)
    try:
        yield budget
    finally:
        budget.close()
        _active.reset(token)
