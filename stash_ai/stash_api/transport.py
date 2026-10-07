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
        self.api = requests.Session()  # No transport-level POST retries; _gql retries only proven read-only operations.
        self.media = requests.Session()
        if self.api_key:
            self.api.headers["ApiKey"] = self.api_key
            self.media.headers["ApiKey"] = self.api_key
        # All read retries share one caller deadline, including media redirects.
        for session in (self.api, self.media):
            adapter = ReadAdapter(max_retries=0)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

    def close(self) -> None:
        self.api.close()
        self.media.close()

    @staticmethod
    def _read_only(query: str, variables: dict | None = None) -> bool:
        if not query.lstrip().startswith('mutation'):
            return True
        from stash_ai.cleanup.sql_session import is_read_only_cleanup
        from stash_ai.stash_api.sql import is_read_only_merge
        return is_read_only_cleanup(query, variables) or is_read_only_merge(query, variables)

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
        read = self._read_only(query, variables)
        if self.dry_run and query.lstrip().startswith("mutation") and not read:
            raise RuntimeError("Internal error: mutation attempted during dry-run.")
        from time import monotonic
        from stash_ai.stash_api.deadlines import call_with_deadline
        from stash_ai.runtime.timing import stage
        report = self.report
        deadline = monotonic() + self._read_budget() if read else None
        response = self._request_response(lambda timeout: self.api.post(
            absolute_stash_url('graphql', self.url),
            json={'query': query, 'variables': variables or {}},
            timeout=timeout, allow_redirects=False, stream=read), read=read, deadline=deadline)
        if report is not None:
            report.response_size(len(response.content))
        response.raise_for_status()
        if 300 <= response.status_code < 400:
            raise RuntimeError("Stash redirected GraphQL; configure STASH_URL to its final URL.")
        with stage(report, 'stash_json_decode'):
            payload = call_with_deadline(response.json, deadline) if read else response.json()
        if payload.get("errors"):
            raise RuntimeError("GraphQL: " + "; ".join(
                error.get("message", "Unknown error") for error in payload["errors"]
            ))
        if not isinstance(payload.get("data"), dict):
            raise RuntimeError("Stash returned no GraphQL data.")
        return payload["data"]

    def _read_budget(self):
        return self.request_timeout[1] if isinstance(self.request_timeout, tuple) else self.request_timeout

    def _request_response(self, call, *, read, deadline, media=False):
        from requests import ConnectionError as RequestsConnectionError, Timeout as RequestsTimeout, Response
        from stash_ai.runtime.timing import stage
        from stash_ai.stash_api.deadlines import ReadDeadlineExceeded, call_with_deadline, remaining, abort_response
        import time
        attempts = self.http_retries + 1 if read else 1
        retry_statuses = {429, 500, 502, 503, 504}
        session = self.media if media else self.api

        def backoff(attempt):
            if self.report is not None:
                self.report.counts['stash_media_retries' if media else 'stash_read_retries'] += 1
            delay = .5 * 2 ** attempt
            budget = remaining(deadline)
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
                timeout = self.request_timeout
            try:
                with stage(self.report, 'stash_media_http' if media else 'stash_http'):
                    response = call_with_deadline(lambda: call(timeout), deadline) if read else call(timeout)
                    if read and isinstance(response, Response):
                        call_with_deadline(lambda: response.content, deadline,
                                           cancel=lambda: abort_response(response))
            except ReadDeadlineExceeded:
                session.close()
                raise
            except (RequestsConnectionError, RequestsTimeout):
                if attempt + 1 == attempts:
                    raise
                backoff(attempt)
                continue
            if not read or response.status_code not in retry_statuses or attempt + 1 == attempts:
                return response
            response.close()
            backoff(attempt)
        raise RuntimeError('Stash request returned no response.')

    def get_media(self, url: str) -> bytes:
        from contextlib import nullcontext
        context = self.report.request('StashAIMediaRead', True) if self.report is not None else nullcontext()
        with context:
            return self._get_media(url)

    def _get_media(self, url):
        from time import monotonic
        deadline = monotonic() + self._read_budget()
        # ApiKey is a custom header; never forward it to another origin.
        for _ in range(6):
            if origin(url) != origin(self.url):
                raise ValueError('Media URL uses a different Stash origin; check STASH_URL.')
            response = self._request_response(lambda timeout: self.media.get(
                url, timeout=timeout, allow_redirects=False, stream=True), read=True, deadline=deadline, media=True)
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
        raise RuntimeError('Too many media redirects.')
