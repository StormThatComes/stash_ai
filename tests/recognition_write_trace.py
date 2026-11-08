"""Check native file events for one offline recognition process tree."""
import csv
from pathlib import Path
import re
import sys


def linux(trace):
    import ast
    pending = {}
    for line in trace.open(errors='replace'):
        pid, _, event = line.strip().partition(' ')
        event = event.lstrip()
        if event.endswith('<unfinished ...>'):
            pending[pid] = event.removesuffix('<unfinished ...>')
            continue
        if event.startswith('<... ') and ' resumed>' in event:
            event = pending.pop(pid,'') + event.split(' resumed>',1)[1]
        if ' = -1' in event:
            continue
        name = event.partition('(')[0]
        if name.startswith('open') and not re.search(r'O_(RDWR|WRONLY|CREAT|TMPFILE|TRUNC)',event):
            continue
        if name not in {'open','openat','creat','mkdir','mkdirat','unlink','unlinkat','rename','renameat','renameat2','link','linkat','symlink','symlinkat','truncate'}:
            continue
        # strace -y annotates directory FDs, including those used by
        # shutil.rmtree to remove children relative to an open directory.
        for field in re.finditer(r'(?:(?:-?\d+|AT_FDCWD)<([^>]+)>,\s*)?("(?:\\.|[^"\\])*")',event):
            path = Path(ast.literal_eval(field[2]))
            if not path.is_absolute() and field[1]:
                path = Path(field[1]) / path
            if not str(path).startswith(('/dev/shm/','/dev/null','/dev/tty')):
                yield path


def windows(trace, root):
    # Procmon records the creator PID on Process Create. Include native
    # descendants such as FFmpeg; unrelated runner services are out of scope.
    pids = {int((root/'probe.pid').read_text())}
    with trace.open(encoding='utf-8-sig',errors='replace',newline='') as stream:
        for row in csv.DictReader(stream):
            if int(row['PID']) not in pids or row['Result']!='SUCCESS':
                continue
            operation, detail, path = row['Operation'], row['Detail'], row['Path']
            if operation=='Process Create':
                match = re.search(r'PID: (\d+)',detail)
                if match:pids.add(int(match[1]))
                continue
            if not re.match(r'^[A-Za-z]:\\',path):
                continue
            # Procmon attributes NTFS bookkeeping to the user request that
            # caused it. These are the volume's internal metadata streams.
            target = Path(path)
            if target.parent == Path(target.anchor) and target.name.casefold() in {
                    '$mft', '$mftmirr', '$bitmap', '$logfile', '$converttononresident'}:
                continue
            if (operation in ('WriteFile','SetRenameInformationFile','SetDispositionInformationFile','SetEndOfFileInformationFile')
                    or operation=='CreateFile' and re.search(r'OpenResult: (Created|Overwritten|Superseded)',detail)):
                yield Path(path)


def verify(root, trace, platform):
    root = root.resolve()
    events = linux(trace) if platform=='linux' else windows(trace,root)
    unexpected, seen, spill = set(), 0, False
    for path in events:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            unexpected.add(str(resolved))
        seen += 1
        spill |= 'etilqs_' in path.name
    if unexpected or not spill or not seen:
        raise SystemExit(f'Native writes={seen}, SQLite spill observed={spill}, outside configured roots={sorted(unexpected)}')
    print(f'Checked {seen} native filesystem events including a SQLite disk spill.')


if __name__=='__main__':
    verify(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
