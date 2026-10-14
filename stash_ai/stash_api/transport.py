"""Stash HTTP transport, read-only retries and credential-safe media access."""
from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse


def absolute_stash_url(path_or_url: str, base_url: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path_or_url)


def origin(url: str) -> tuple[str, str | None, int]:
    parsed = urlparse(url)
    return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


def safe_error(exc: Exception, api_key: str = "") -> str:
    """Avoid exposing API keys embedded in Stash's media URLs or HTTP errors."""
    message = str(exc)
    if api_key:
        message = message.replace(api_key, "[redacted]")
    message = re.sub(r'(?i)((?:https?:)?//)[^/?#\"\'<>]*@', r'\1', message)
    message = re.sub(r'(?i)(netloc\s+[\"\'])[^\"\']*@', r'\1', message)
    return re.sub(r"(?i)([?&](?:apikey|api_key|token)=)[^&\s\"'<>]+", r"\1[redacted]", message)


class StashTransport:
    """Own HTTP sessions and enforce GraphQL write/retry and redirect policies."""

    def __init__(self, dry_run: bool, *, url: str, api_key: str,
                 request_timeout: float | tuple[float, float], http_retries: int):
        import requests
        from stash_ai.stash_api.read_connections import ReadAdapter

        self.url = url
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.http_retries = http_retries
        self.dry_run = dry_run
        self.report = None
        self.api = self.media = None
        pending_adapter = None
        try:
            # No transport-level POST retries; _gql retries proven reads only.
            self.api = requests.Session()
            self.media = requests.Session()
            if self.api_key:
                self.api.headers["ApiKey"] = self.api_key
                self.media.headers["ApiKey"] = self.api_key
            # Sessions own each adapter only after both mounts succeed.
            for session in (self.api, self.media):
                pending_adapter = ReadAdapter(max_retries=0)
                session.mount("http://", pending_adapter)
                session.mount("https://", pending_adapter)
                pending_adapter = None
        except BaseException:
            from stash_ai.runtime.cleanup import cleanup
            actions = []
            if pending_adapter is not None:
                def detach():
                    for session in (self.api, self.media):
                        mounted = getattr(session, 'adapters', None)
                        if isinstance(mounted, dict):
                            for prefix, adapter in tuple(mounted.items()):
                                if adapter is pending_adapter:
                                    del mounted[prefix]
                actions = [('detach incomplete HTTP adapter', detach),
                           ('close incomplete HTTP adapter', pending_adapter.close)]
            cleanup(actions + [('close partially initialized HTTP client', self.close)])
            raise

    def close(self) -> None:
        from stash_ai.runtime.cleanup import cleanup
        cleanup([(name, session.close) for name, session in
                 (('close HTTP API session', self.api), ('close HTTP media session', self.media))
                 if session is not None], report=self.report)

    def new_media_reader(self):
        """Give background input reads their own sessions and deadline cleanup."""
        reader = StashTransport(True, url=self.url, api_key=self.api_key,
                                request_timeout=self.request_timeout,
                                http_retries=self.http_retries)
        reader.report = self.report
        return reader

    @staticmethod
    def _read_only(query: str, variables: dict | None = None) -> bool:
        from stash_ai.cleanup.sql_session import is_read_only_cleanup
        from stash_ai.stash_api.sql import is_read_only_merge
        if is_read_only_cleanup(query, variables) or is_read_only_merge(query, variables):
            return True
        # Ignore comments and string contents, but inspect the whole document:
        # a query followed by a mutation is not a proven read either.
        tokens = re.findall(r'"""(?:\\.|(?!""")[\s\S])*"""|"(?:\\.|[^"\\])*"|#[^\r\n]*|[A-Za-z_][A-Za-z_0-9]*|[{}()]', query)
        depth = 0
        operations = []
        for token in tokens:
            if token.startswith(('"', '#')):
                continue
            if depth == 0 and token in ('query', 'mutation', 'subscription'):
                operations.append(token)
            if token in ('{', '('):
                depth += 1
            elif token in ('}', ')'):
                depth -= 1
        first = re.sub(r'#[^\r\n]*', '', query).lstrip('\ufeff \t\r\n,')
        return depth == 0 and (bool(operations) and all(op == 'query' for op in operations)
                               or not operations and first.startswith('{'))

    def gql(self, query: str, variables: dict | None = None) -> dict:
        report = self.report
        if report is None:
            return self._gql(query, variables)
        read = self._read_only(query, variables)
        # Names are from our fixed operations, never from variables or payloads.
        match = re.search(r'\b(?:query|mutation)\s+(StashAI[A-Za-z0-9_]*)\b', query)
        operation = match.group(1) if match else ('read' if read else 'write')
        if operation == 'StashAIMergeReadSQL':
            from stash_ai.stash_api.sql import SQL
            operation += '.' + next((mode for mode, sql in SQL.items() if sql == (variables or {}).get('sql')), 'unknown')
        if operation == 'StashAICleanupReadSQL':
            from stash_ai.cleanup.sql_session import SQL
            operation += '.' + next((mode for mode, sql in SQL.items() if sql == (variables or {}).get('sql')), 'unknown')
        with report.request(operation, read):
            return self._gql(query, variables)

    def _gql(self, query: str, variables: dict | None = None) -> dict:
        from time import monotonic
        from stash_ai.stash_api.deadlines import call_with_deadline, ReadDeadlineExceeded
        read = self._read_only(query, variables)
        # Never schedule or time a denied mutation, including in lightweight
        # dry-run/test clients without a configured network timeout.
        if self.dry_run and not read:
            error = RuntimeError("Internal error: mutation attempted during dry-run.")
            error.write_outcome = 'failed'
            raise error
        # One elapsed boundary covers dispatch, response body and JSON. Mutation
        # dispatch has its own gate and never participates in read retry policy.
        deadline = monotonic() + self._read_budget()
        from stash_ai.stash_api.attempts import MutationAttempt
        attempt = None if read else MutationAttempt()
        operation = lambda: self._gql_payload(query, variables, read=read, deadline=deadline, attempt=attempt)
        if self.report is not None:
            operation = self.report.bind_read(operation)
        if not read:
            try:
                return call_with_deadline(operation, deadline, label='mutation')
            except ReadDeadlineExceeded as exc:
                dispatched = attempt.expire()
                error = RuntimeError('Stash mutation timed out; write outcome is unknown. '
                                     'Inspect/recover pending publication before retrying.' if dispatched else
                                     'Stash mutation expired before dispatch; no write was sent.')
                error.write_outcome = 'unknown' if dispatched else 'failed'
                raise error from exc
            except BaseException as exc:
                dispatched = attempt.expire()
                from requests.exceptions import ConnectTimeout, InvalidURL, InvalidSchema, MissingSchema
                if not dispatched or isinstance(exc, (ConnectTimeout, InvalidURL, InvalidSchema, MissingSchema)):
                    exc.write_outcome = 'failed'
                raise
        try:
            return call_with_deadline(operation, deadline)
        except (ReadDeadlineExceeded, KeyboardInterrupt):
            from stash_ai.runtime.cleanup import cleanup
            cleanup([self.api.close], report=self.report)
            raise

    def _gql_payload(self, query, variables, *, read, deadline, attempt=None):
        if self.dry_run and not read:
            raise RuntimeError("Internal error: mutation attempted during dry-run.")
        from stash_ai.stash_api.deadlines import call_with_deadline
        from stash_ai.runtime.timing import stage
        report = self.report
        response = self._request_response(lambda timeout: self.api.post(
            absolute_stash_url('graphql', self.url),
            json={'query': query, 'variables': variables or {}},
            timeout=timeout, allow_redirects=False, stream=True), read=read, deadline=deadline, mutation=attempt)
        try:
            if report is not None:
                report.response_size(len(response.content))
            try:
                response.raise_for_status()
            except Exception as exc:
                if not read and response.status_code in (401, 403):
                    exc.write_outcome = 'failed'
                if response.status_code == 400:
                    # Stash returns variable/schema validation details in the
                    # JSON body of HTTP 400. Keep HTTPError compatibility while
                    # exposing bounded, credential-redacted messages, not the
                    # complete response or submitted variables.
                    try:
                        rejected = call_with_deadline(response.json, deadline)
                    except Exception:
                        rejected = None
                    errors = rejected.get('errors') if isinstance(rejected, dict) else None
                    if (isinstance(errors, list) and errors and all(
                            isinstance(item, dict) and isinstance(item.get('message'), str) for item in errors)):
                        from requests import HTTPError
                        messages = [safe_error(item['message'], self.api_key)[:400] for item in errors[:8]]
                        error = HTTPError(safe_error(exc, self.api_key) + '; GraphQL: ' + '; '.join(messages)[:1600],
                                          response=response, request=getattr(response, 'request', None))
                        if not read and not rejected.get('data') and all(
                                isinstance(item.get('extensions'), dict) and item['extensions'].get('code')
                                in ('GRAPHQL_PARSE_FAILED', 'GRAPHQL_VALIDATION_FAILED') for item in errors):
                            error.write_outcome = 'failed'
                        raise error from exc
                raise
            if 300 <= response.status_code < 400:
                raise RuntimeError("Stash redirected GraphQL; configure STASH_URL to its final URL.")
            with stage(report, 'stash_json_decode'):
                payload = call_with_deadline(response.json, deadline)
            if not isinstance(payload, dict):
                raise RuntimeError('Invalid GraphQL response envelope: expected object.')
            errors = payload.get('errors')
            if errors is not None and (not isinstance(errors, list) or
                    any(not isinstance(item, dict) or not isinstance(item.get('message'), str)
                        for item in errors)):
                raise RuntimeError('Invalid GraphQL errors envelope.')
            if errors:
                error = RuntimeError("GraphQL: " + "; ".join(error["message"] for error in errors))
                if not read and not payload.get('data') and all(
                        isinstance(item.get('extensions'), dict) and item['extensions'].get('code')
                        in ('GRAPHQL_PARSE_FAILED', 'GRAPHQL_VALIDATION_FAILED') for item in errors):
                    error.write_outcome = 'failed'
                raise error
            if not isinstance(payload.get("data"), dict):
                raise RuntimeError("Stash returned no GraphQL data.")
            return payload["data"]
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                from stash_ai.runtime.cleanup import cleanup
                cleanup([close], report=report)

    def _read_budget(self):
        return self.request_timeout[1] if isinstance(self.request_timeout, tuple) else self.request_timeout

    def _request_response(self, call, *, read, deadline, media=False, mutation=None):
        from requests import ConnectionError as RequestsConnectionError, Timeout as RequestsTimeout, Response
        from stash_ai.runtime.timing import stage
        from stash_ai.stash_api.deadlines import ReadDeadlineExceeded, call_with_deadline, remaining, abort_response
        import time
        attempts = self.http_retries + 1 if read else 1
        retry_statuses = {429, 500, 502, 503, 504}
        session = self.media if media else self.api

        def event(**values):
            callback = getattr(self.report, 'request_state', None)
            if callback is not None:
                self.report.optional('request_status', callback, **values)

        def backoff(attempt, reason):
            if self.report is not None:
                self.report.counts['stash_media_retries' if media else 'stash_read_retries'] += 1
            delay = .5 * 2 ** attempt
            budget = remaining(deadline)
            event(state='backoff', attempt=attempt+1, attempts=attempts,
                  delay=min(delay, budget), budget=budget, reason=reason)
            from stash_ai.runtime.cancellation import current
            token = current()
            if token is not None:
                token.event.wait(min(delay, budget))
                token.check()
            else:
                time.sleep(min(delay, budget))
            remaining(deadline)

        for attempt in range(attempts):
            if read:
                budget = remaining(deadline)
                timeout = ((min(self.request_timeout[0], budget), budget)
                           if isinstance(self.request_timeout, tuple) else budget)
            else:
                budget = remaining(deadline)
                timeout = ((min(self.request_timeout[0], budget), budget)
                           if isinstance(self.request_timeout, tuple) else budget)
            response = None
            event(state='request', attempt=attempt+1, attempts=attempts, budget=budget)
            try:
                with stage(self.report, 'stash_media_http' if media else 'stash_http'):
                    response = (call_with_deadline(lambda: call(timeout), deadline) if read else
                                mutation.dispatch(call, timeout, deadline) if mutation is not None else call(timeout))
                    if isinstance(response, Response):
                        # The nested body read uses the enclosing owned operation.
                        call_with_deadline(lambda: response.content, deadline,
                                           cancel=lambda: abort_response(response))
                remaining(deadline)
                event(state='response', http_status=response.status_code)
            except (ReadDeadlineExceeded, KeyboardInterrupt):
                from stash_ai.runtime.cleanup import cleanup
                cleanup([response.close] if response is not None else [], report=self.report)
                cleanup([session.close], report=self.report)
                raise
            except (RequestsConnectionError, RequestsTimeout) as exc:
                if response is not None:
                    from stash_ai.runtime.cleanup import cleanup
                    cleanup([response.close], report=self.report)
                if attempt + 1 == attempts:
                    event(state='failed', reason=type(exc).__name__)
                    raise
                backoff(attempt, type(exc).__name__)
                continue
            except BaseException:
                from stash_ai.runtime.cleanup import cleanup
                cleanup([response.close] if response is not None else [], report=self.report)
                raise
            if not read or response.status_code not in retry_statuses or attempt + 1 == attempts:
                return response
            # A cleanup fault cannot consume an otherwise safe read retry.
            # Preserve the discarded HTTP outcome as the primary event.
            from requests import HTTPError
            from stash_ai.runtime.cleanup import cleanup
            cleanup([response.close], report=self.report,
                    primary=HTTPError('Discarded retryable HTTP response.', response=response))
            backoff(attempt, 'HTTP' + str(response.status_code))
        raise RuntimeError('Stash request returned no response.')

    def get_media(self, url: str) -> bytes:
        from contextlib import nullcontext
        context = self.report.request('StashAIMediaRead', True) if self.report is not None else nullcontext()
        with context:
            return self._get_media(url)

    def _get_media(self, url):
        from time import monotonic
        from stash_ai.stash_api.deadlines import call_with_deadline, ReadDeadlineExceeded
        deadline = monotonic() + self._read_budget()
        operation = lambda: self._media_payload(url, deadline)
        if self.report is not None:
            operation = self.report.bind_read(operation)
        try:
            return call_with_deadline(operation, deadline)
        except (ReadDeadlineExceeded, KeyboardInterrupt):
            from stash_ai.runtime.cleanup import cleanup
            cleanup([self.media.close], report=self.report)
            raise

    def _media_payload(self, url, deadline):
        # ApiKey is a custom header; never forward it to another origin.
        for _ in range(6):
            if origin(url) != origin(self.url):
                raise ValueError('Media URL uses a different Stash origin; check STASH_URL.')
            response = self._request_response(lambda timeout: self.media.get(
                url, timeout=timeout, allow_redirects=False, stream=True), read=True, deadline=deadline, media=True)
            try:
                if self.report is not None:
                    self.report.response_size(len(response.content))
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get('Location')
                    if not location:
                        raise RuntimeError('Media redirect has no Location header.')
                    url = urljoin(url, location)
                    continue
                response.raise_for_status()
                return response.content
            finally:
                close = getattr(response, "close", None)
                if close is not None:
                    from stash_ai.runtime.cleanup import cleanup
                    cleanup([close], report=self.report)
        raise RuntimeError('Too many media redirects.')
