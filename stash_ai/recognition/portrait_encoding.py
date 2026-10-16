"""One ordered, byte-bounded CPU encoding job; no database, HTTP or models."""
from concurrent.futures import ThreadPoolExecutor
from stash_ai.runtime.cleanup import cleanup, closing
from .portrait_threads import encoding_workers


class Encoder:
    def __init__(self,config,crop_function,*,small=False):
        self.function=crop_function
        self.workers=0 if small or not config.data['execution'].get('prefetch_portraits',True) else encoding_workers(config)
        self.limit=min(67108864,config.data['execution']['rebuild_memory_bytes']//32)
        self.executor=None

    def __enter__(self):return self

    def call(self,function,*args,size,before_wait=lambda:None):
        if not self.workers or size>self.limit:
            before_wait()
            return function(*args)
        if self.executor is None:
            self.executor=ThreadPoolExecutor(max_workers=self.workers,thread_name_prefix='portrait-jpeg')
        future=self.executor.submit(function,*args)
        try:
            before_wait()
            return future.result()
        except BaseException:
            future.cancel();raise

    def encode(self,image,face,policy,limit,*,before_wait=lambda:None):
        def encode():
            with closing(image,name='release portrait encoding pixels'):
                details={}
                data,bounds=self.function(image,face,policy,limit=limit,details=details)
                return data,bounds,details
        size=image.width*image.height*3+limit
        if not self.workers or size>self.limit:
            try:before_wait()
            except BaseException:
                image.close();raise
            return encode()
        try:
            if self.executor is None:
                self.executor=ThreadPoolExecutor(max_workers=self.workers,thread_name_prefix='portrait-jpeg')
            future=self.executor.submit(encode)
        except BaseException:
            image.close();raise
        future.add_done_callback(lambda f:image.close() if f.cancelled() else None)
        try:
            before_wait()
            return future.result()
        except BaseException:
            future.cancel();raise

    def __exit__(self,*error):
        if self.executor is not None:
            cleanup([('join portrait JPEG encoder',lambda:self.executor.shutdown(wait=True,cancel_futures=True))],
                    primary=error[1])
