"""Codec telemetry reaches both optional diagnostics and ordinary run counters."""
from stash_ai.runtime.diagnostics import MergeDiagnostics
from stash_ai.runtime.report import RunReport
from stash_ai.storage.codec import decode, encode


def test_byte_counters_are_not_doubled_by_merge_diagnostics():
    report = RunReport(write=lambda _: None)
    with MergeDiagnostics(report) as diagnostic:
        blob = encode({'value': 'data'}, report)
        assert decode(blob, report) == {'value': 'data'}
        assert report.counts == diagnostic.counts
        assert report.counts['state_blob_bytes_written'] == len(blob)
        assert report.counts['state_blob_bytes_read'] == len(blob)
