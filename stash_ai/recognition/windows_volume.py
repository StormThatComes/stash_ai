"""Windows volume identities for the disk budget and its cross-process leases."""
import ctypes
from ctypes import wintypes
from nt import _getvolumepathname
import ntpath


_volume_name = ctypes.WinDLL('kernel32', use_last_error=True).GetVolumeNameForVolumeMountPointW
_volume_name.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
_volume_name.restype = wintypes.BOOL


def volume_key(path):
    # Resolve mapped drives and junctions before locating the actual mount.
    mount = ntpath.normcase(_getvolumepathname(ntpath.realpath(path))).rstrip('\\') + '\\'
    if mount.startswith('\\\\?\\unc\\'):
        mount = '\\\\' + mount[8:]
    if mount.startswith('\\\\') and not mount.startswith(('\\\\?\\', '\\\\.\\')):
        return mount  # SMB shares have no local volume GUID.

    name = ctypes.create_unicode_buffer(51)
    if _volume_name(mount, name, len(name)):
        return name.value.casefold()
    error = ctypes.get_last_error()
    if error in (1, 50):  # ERROR_INVALID_FUNCTION, ERROR_NOT_SUPPORTED
        return mount
    raise ctypes.WinError(error)
