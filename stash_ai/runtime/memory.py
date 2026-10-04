"""Resident memory and retained-object estimates with stdlib-only fallbacks."""
import sys


def process_rss():
    """Current resident memory on Windows/Linux; no peak-RSS approximation."""
    try:
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ('peak', 'working', 'paged_peak',
                    'paged', 'nonpaged_peak', 'nonpaged', 'pagefile', 'pagefile_peak')]
            result = Counters()
            result.cb = ctypes.sizeof(result)
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL('psapi', use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(result), result.cb):
                return result.working
        elif sys.platform.startswith('linux'):
            import os
            with open('/proc/self/statm') as stream:
                return int(stream.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    return None


def retained_size(value):
    """Count retained Python objects once, without serializing a second copy."""
    pending, seen, total = [value], set(), 0
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        total += sys.getsizeof(item)
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            pending.extend(item)
        if hasattr(item, '__dict__'):
            pending.append(vars(item))
    return total
