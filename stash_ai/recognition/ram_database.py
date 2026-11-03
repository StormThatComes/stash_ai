"""Owner-thread RAM SQLite writers sharing the rebuild's reservation budget.

Page limits are set before writes and page ownership is refreshed afterwards.
Read-only lookahead connections need no reservation: they share these pages.
Temporary SQLite work and native/Python overhead are not an OS RSS guarantee.
"""
import sqlite3


class Connection(sqlite3.Connection):
    budget = None

    def attach_budget(self, budget, label):
        self.budget = budget
        self.owner = (label, id(self))
        self.page_size = super().execute('PRAGMA main.page_size').fetchone()[0]
        budget.register_database(self)

    def set_capacity(self, size):
        pages = max(1, size // self.page_size)
        if pages != getattr(self, '_capacity_pages', None):
            super().execute(f'PRAGMA main.max_page_count={pages}')
            self._capacity_pages = pages

    def account_pages(self):
        size = super().execute('PRAGMA main.page_count').fetchone()[0] * self.page_size
        self.budget.reserve(self.owner, size)

    def _call(self, function, sql, *args, **kwargs):
        # SELECT/EXPLAIN cannot grow the main database. PRAGMA and WITH are
        # deliberately included: either can write. Cursor APIs use this too.
        read = sql.lstrip().upper().startswith(('SELECT ', 'EXPLAIN '))
        if self.budget is None or read:
            return function(sql, *args, **kwargs)
        try:
            return function(sql, *args, **kwargs)
        except sqlite3.OperationalError as error:
            if getattr(error, 'sqlite_errorcode', None) == sqlite3.SQLITE_FULL:
                requested = self.budget.working + self.page_size
                raise self.budget.resource_error(self.owner, requested, minimum=True) from error
            raise
        finally:
            self.account_pages()

    def execute(self, sql, *args, **kwargs):
        return self._call(super().execute, sql, *args, **kwargs)

    def executemany(self, sql, *args, **kwargs):
        return self._call(super().executemany, sql, *args, **kwargs)

    def executescript(self, sql, *args, **kwargs):
        return self._call(super().executescript, sql, *args, **kwargs)

    def cursor(self, factory=None):
        return super().cursor(factory or Cursor)

    def rollback(self):
        try:
            return super().rollback()
        finally:
            if self.budget is not None:
                self.account_pages()

    def __exit__(self, *error):
        try:
            return super().__exit__(*error)
        finally:
            if self.budget is not None:
                self.account_pages()

    def close(self):
        budget, self.budget = self.budget, None
        try:
            super().close()
        finally:
            if budget is not None:
                budget.release_database(self)


class Cursor(sqlite3.Cursor):
    def execute(self, sql, *args, **kwargs):
        return self.connection._call(super().execute, sql, *args, **kwargs)

    def executemany(self, sql, *args, **kwargs):
        return self.connection._call(super().executemany, sql, *args, **kwargs)

    def executescript(self, sql, *args, **kwargs):
        return self.connection._call(super().executescript, sql, *args, **kwargs)


def connect(config, uri, *, label):
    from .run_cache import current, Cache
    from .resource_plan import memory_limit
    budget = current()
    if budget is None:
        budget = Cache(0, 0)
        budget.configure_working(memory_limit(config), 0)
    db = sqlite3.connect(uri, uri=True, factory=Connection)
    try:
        db.attach_budget(budget, label)
    except BaseException:
        db.close()
        raise
    return db
