"""Controlled worker overhead and zero-cache verification memory benchmark."""
import argparse
import json
import time
import tracemalloc
from types import SimpleNamespace

import requests

from stash_ai.inference.inputs import RunMedia
from stash_ai.runtime.cache import RamCache
from stash_ai.stash_api import deadlines
from stash_ai.stash_api.transport import StashTransport


def reads(count):
    client = StashTransport(False, url='http://stash.test', api_key='', request_timeout=60., http_retries=0)
    def response(*args, **kwargs):
        result = requests.Response()
        result.status_code, result._content = 200, b'{"data":{"ping":true}}'
        result.close = lambda: None
        return result
    client.api.post = response
    original, result = deadlines.Thread, {}
    try:
        for mode in ('separate_stages', 'owned_operation'):
            starts = []
            def thread(*args, **kwargs):
                if kwargs.get('name') == 'stash-deadline-read': starts.append(1)
                return original(*args, **kwargs)
            deadlines.Thread = thread
            started = time.perf_counter()
            for _ in range(count):
                if mode == 'separate_stages':
                    client._gql_payload('query StashAIPing { ping }', None, read=True, deadline=time.monotonic()+60.)
                else:
                    client.gql('query StashAIPing { ping }')
            elapsed = time.perf_counter()-started
            result[mode] = dict(reads=count, workers=len(starts), seconds=elapsed, reads_per_second=count/elapsed)
    finally:
        deadlines.Thread = original
        client.close()
    return result


def hashes(count):
    import hashlib
    result = {}
    for mode in ('dictionary', 'sqlite_ledger'):
        tracemalloc.start()
        started = time.perf_counter()
        media = None
        if mode == 'dictionary':
            values = {}
            for i in range(count):
                url = f'http://stash.test/sprites/{i}'
                values[url] = hashlib.sha256(url.encode()).hexdigest()
        else:
            media = RunMedia(SimpleNamespace(get_media=lambda url: url.encode()), RamCache(0))
            for i in range(count): media.get_media(f'http://stash.test/sprites/{i}')
        elapsed, (_, peak) = time.perf_counter()-started, tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result[mode] = dict(entries=count, seconds=elapsed, python_peak_bytes=peak)
        if media is not None:
            db = media.hashes._db
            result[mode].update(page_cache_kib=64, temporary_db_bytes=db.execute('PRAGMA page_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0])
            media.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reads', type=int, default=200)
    parser.add_argument('--urls', type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(dict(reads=reads(args.reads), verification=hashes(args.urls)), indent=2))


if __name__ == '__main__':
    main()
