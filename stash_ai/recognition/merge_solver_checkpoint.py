"""Durable completed-trial journal and owned atomic outputs."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .config import digest, plain


def sha(content):
    return hashlib.sha256(content).hexdigest()


class AtomicFile:
    def __init__(self,path,expected=None):
        self.path=Path(path).expanduser().resolve()
        self.expected=expected
        self.check()

    def check(self):
        actual=sha(self.path.read_bytes()) if self.path.exists() else None
        if actual!=self.expected:
            raise FileExistsError(f'Output already exists or changed outside this solver: {self.path}')

    def write(self,text):
        content=text.encode()
        self.check()
        if self.expected==sha(content):return
        fd,name=tempfile.mkstemp(prefix='.'+self.path.name+'-',dir=self.path.parent)
        try:
            with os.fdopen(fd,'wb') as stream:
                stream.write(content);stream.flush();os.fsync(stream.fileno())
            self.check()
            if self.expected is None:
                os.link(name,self.path)  # Never replace a concurrently created file.
            else:
                os.replace(name,self.path)
            self.expected=sha(content)
            if os.name!='nt':
                directory=os.open(self.path.parent,os.O_RDONLY)
                try:os.fsync(directory)
                finally:os.close(directory)
        finally:
            # SIGINT may arrive immediately after the atomic rename.
            if self.path.exists() and sha(self.path.read_bytes())==sha(content):
                self.expected=sha(content)
            if os.path.exists(name):os.unlink(name)


def fingerprint(config,goals,db,settings):
    """Replay inputs, policy, goals, code and search settings define compatibility."""
    from .replay_signature import fingerprint as source_fingerprint
    source=source_fingerprint(db,goals['scope'],goals['namespace'])
    code=hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        code.update(path.name.encode());code.update(path.read_bytes())
    return dict(format=2,source=source,code=code.hexdigest(),
                configuration=digest(plain(config.data)),goals=digest(goals),settings=settings,
                database=str(config.database.resolve()),config_path=str(config.source_path))


class Journal:
    def __init__(self,path,signature,*,resume=False,winner=None):
        from contextlib import ExitStack
        from stash_ai.storage.writer_lock import acquire_writer_lock
        self._locks = ExitStack()
        self._ready = False
        paths = {Path(path).expanduser().resolve()}
        if winner is not None:paths.add(winner.path)
        try:
            for destination in sorted(paths):
                try:
                    self._locks.enter_context(acquire_writer_lock(destination))
                except RuntimeError as error:
                    raise RuntimeError(f'Another solver owns checkpoint/output: {destination}') from error
            self._initialize(path, signature, resume=resume, winner=winner)
            self._ready = True
            from .solver_journal_events import _journals
            owners = _journals.get()
            if owners is not None:owners.append(self)
        except BaseException:
            self._locks.close()
            raise

    def close(self):
        try:
            if self._ready:self.flush_snapshot()
        finally:
            self._ready = False
            self._locks.close()

    def __del__(self):
        # Crash/abandoned standalone owner: events are already durable. Do not
        # publish snapshots from a finalizer, but always release OS ownership.
        if hasattr(self, '_locks'):self._locks.close()

    def _initialize(self,path,signature,*,resume=False,winner=None):
        self.path=Path(path).expanduser().resolve()
        self.directory=self.path.with_name(self.path.name+'.trials')
        self.records={}
        self.winner=winner
        if resume:
            raw=self.path.read_bytes()
            from .solver_journal_events import read
            self.document, sequence, previous, valid_bytes = read(self.path)
            if self.document['signature']!=signature:
                raise ValueError('Solver checkpoint is incompatible with retained inputs, config, goals, code or search settings.')
            self.file=AtomicFile(self.path,sha(raw))
            saved=self.document.get('winner')
            if saved is None and winner is not None and winner.expected is not None:
                raise FileExistsError('Winner YAML is not owned by this checkpoint.')
            if saved:
                if winner is None or str(winner.path)!=saved['path']:
                    raise ValueError('Resume requires the original winner YAML path.')
                actual=sha(winner.path.read_bytes()) if winner.path.exists() else None
                if actual not in (saved['sha256'],saved['previous_sha256']):
                    raise FileExistsError('Winner YAML changed outside this solver.')
                winner.expected=actual
                winner.write(saved['text'])
            for entry in self.document['records']:
                raw=(self.directory/entry['file']).read_bytes()
                if sha(raw)!=entry['sha256']:raise ValueError('Corrupt solver checkpoint trial.')
                item=json.loads(raw);self.records[item['name']]=entry
        else:
            self.file=AtomicFile(self.path)
            self.directory.mkdir()  # Existing journals are never silently overwritten.
            self.document=dict(signature=signature,records=[],winner=None)
            from .solver_journal_events import Events
            self.events = Events(self)
            self.file.write(json.dumps(self.document,allow_nan=False))
        if resume:
            from .solver_journal_events import Events
            self.events = Events(self, resume=True, state=(sequence,previous,valid_bytes) if self.document.get('journal_format') == 1 else None)

    def flush_snapshot(self):
        self.document['journal_applied'] = self.events.sequence
        self.file.write(json.dumps(self.document, allow_nan=False))

    def get(self,name):
        entry=self.records[name]
        raw=(self.directory/entry['file']).read_bytes()
        if sha(raw)!=entry['sha256']:raise ValueError('Corrupt solver checkpoint trial.')
        return json.loads(raw)

    def schedule(self, kind, name):
        self.document.setdefault('schedule', []).append(dict(kind=kind, name=name))
        self.events.append(dict(kind='schedule', entry=dict(kind=kind, name=name)))

    def save_raw_result(self, name, settings, result):
        content = json.dumps(result, allow_nan=False)
        filename = name + '-raw-' + sha(content.encode()) + '.json'
        path = self.directory / filename
        AtomicFile(path, sha(path.read_bytes()) if path.exists() else None).write(content)
        self.document.setdefault('raw_results', {})[name] = dict(file=filename,
            sha256=sha(content.encode()), configuration=digest(plain(settings.data)))
        self.events.append(dict(kind='raw', name=name, entry=self.document['raw_results'][name]))

    def raw_result(self, name, settings=None):
        row = self.document.get('raw_results', {}).get(name)
        if row is None:
            return None
        if settings is not None and row['configuration'] != digest(plain(settings.data)):
            raise ValueError('Checkpoint raw result configuration changed.')
        # Paths are coordinator-generated basenames, never arbitrary destinations.
        if Path(row['file']).name != row['file']:
            raise ValueError('Invalid checkpoint result path.')
        content = (self.directory / row['file']).read_bytes()
        if sha(content) != row['sha256']:
            raise ValueError('Corrupt completed trial result.')
        return json.loads(content)

    def task(self, name, settings, status):
        if status not in ('queued', 'running', 'completed', 'failed', 'interrupted'):
            raise ValueError('Invalid trial lifecycle state.')
        tasks = self.document.setdefault('tasks', {})
        configuration = digest(plain(settings.data))
        previous = tasks.get(name)
        if previous and previous['configuration'] != configuration:
            raise ValueError('Checkpoint pending trial configuration changed.')
        tasks[name] = dict(configuration=configuration, status=status)
        self.events.append(dict(kind='task', name=name, entry=tasks[name]))

    def record(self,name,settings,item,state,winner_text):
        record=dict(name=name,configuration=digest(plain(settings.data)),item=item,state=state)
        text=json.dumps(record,allow_nan=False)
        filename=name+'-'+sha(text.encode())+'.json'
        path=self.directory/filename
        # A crash may leave an uncommitted record; only this journal owns the directory.
        AtomicFile(path,sha(path.read_bytes()) if path.exists() else None).write(text)
        entry = dict(file=filename,sha256=sha(text.encode()))
        self.document['records'].append(entry)
        self.records[name] = entry
        if name in self.document.get('tasks', {}):
            self.document['tasks'][name]['status'] = 'failed' if item.get('failed') else 'completed'
        if self.winner and winner_text is not None:
            self.document['winner']=dict(path=str(self.winner.path),text=winner_text,
                sha256=sha(winner_text.encode()),previous_sha256=self.winner.expected)
        raw = self.document.get('raw_results', {}).pop(name, None)
        self.events.append(dict(kind='record', name=name, entry=entry,
            task=self.document.get('tasks', {}).get(name),
            winner=self.document['winner'] if self.winner and winner_text is not None else None))
        # The applied record is durable before its superseded raw result is removed.
        if raw is not None and Path(raw['file']).name == raw['file']:
            (self.directory/raw['file']).unlink(missing_ok=True)
        if self.winner and winner_text is not None:self.winner.write(winner_text)


def winner_file(path,*,resume=False):
    if path is None:return None
    path=Path(path).expanduser().resolve()
    # Journal verification is mandatory before this provisional owner can write.
    return AtomicFile(path,sha(path.read_bytes()) if resume and path.exists() else None)
