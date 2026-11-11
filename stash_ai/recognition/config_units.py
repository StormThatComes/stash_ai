"""Byte bounds shared by legacy settings and public fractional MB limits."""
from stash_ai.byte_units import BYTES_PER_MB, mb_to_bytes

BYTE_LIMITS = {
    'execution.refinement_queue_bytes': (1048576, 4294967296),
    'execution.retained_memory_bytes': (1048576, 1099511627776),
    'execution.model_bytes_cache_bytes': (0, 8589934592),
    'clustering.max_reference_bytes': (1048576, 68719476736),
    'portraits.max_video_bytes': (1048576, 1073741824),
    'execution.disk_total_bytes': (0, 10995116277760),
    'execution.disk_free_bytes': (0, 1099511627776),
    'execution.rebuild_sqlite_cache_bytes': (32768, 1073741824),
    'execution.rebuild_scratch_bytes': (1048576, 1099511627776),
    'execution.rebuild_queue_bytes': (65536, 1073741824),
    'execution.cpu_residency_bytes': (0, 68719476736),
    'execution.scan_queue_bytes': (65536, 4294967296),
    'execution.inference_cache_bytes': (0, 68719476736),
    'execution.sprite_cache_bytes': (0, 68719476736),
    'execution.media_cache_bytes': (0, 68719476736),
    'execution.portrait_ranking_cache_bytes': (0, 68719476736),
    'execution.derived_cache_bytes': (0, 1099511627776),
    'execution.pixel_index_bytes': (1048576, 1099511627776),
    'execution.rebuild_cache_bytes': (0, 68719476736),
    'execution.rebuild_matrix_cache_bytes': (0, 68719476736),
    'execution.rebuild_checkpoint_bytes': (1048576, 8589934592),
    'execution.rebuild_memory_bytes': (1048576, 1099511627776),
    'execution.replay_snapshot_bytes': (1048576, 1099511627776),
    'storage.max_crop_bytes': (1024, 8388608),
    'sapiensid.execution.reference_bytes': (1, None),
    'sapiensid.execution.plan_bytes': (1, None),
}


def from_mb(path, value, *, name):
    low, high = BYTE_LIMITS[path]
    return mb_to_bytes(value, name=name, minimum=low / BYTES_PER_MB,
                       maximum=None if high is None else high / BYTES_PER_MB)
