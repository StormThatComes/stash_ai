"""Best-effort available RAM, respecting container limits when exposed."""
from pathlib import Path
import os


def available():
    bounds = []
    if os.name == 'nt':
        import ctypes
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total',
                    'page_available', 'virtual_total', 'virtual_available', 'extended')]
        status = Status(); status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            bounds.append(status.available)
    else:
        try:
            fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            bounds.append(int(fields['MemAvailable'].split()[0]) * 1024)
        except (OSError, KeyError, ValueError):
            try:
                bounds.append(os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
            except (ValueError, OSError, AttributeError):
                pass
        for directory, limit, used in (
            ('/sys/fs/cgroup', 'memory.max', 'memory.current'),
            ('/sys/fs/cgroup/memory', 'memory.limit_in_bytes', 'memory.usage_in_bytes')):
            try:
                ceiling = int((Path(directory)/limit).read_text())
                bounds.append(max(0, ceiling-int((Path(directory)/used).read_text())))
            except (OSError, ValueError):
                pass
    return min(bounds) if bounds else None
