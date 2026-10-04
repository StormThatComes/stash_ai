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
                 request_timeout: float, http_retries: int):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

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
        adapter = HTTPAdapter(max_retries=Retry(
            total=http_retries, backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        ))
        self.media.mount("http://", adapter)
        self.media.mount("https://", adapter)

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
        with report.request(operation, read):
            return self._gql(query, variables)

    def _gql(self, query: str, variables: dict | None = None) -> dict:
        read = self._read_only(query, variables)
        if self.dry_run and query.lstrip().startswith("mutation") and not read:
            raise RuntimeError("Internal error: mutation attempted during dry-run.")
        from requests import ConnectionError as RequestsConnectionError, Timeout as RequestsTimeout
        from stash_ai.runtime.timing import stage
        import time
        report = self.report
        attempts = self.http_retries + 1 if read else 1
        retry_statuses = {429, 500, 502, 503, 504}
        for attempt in range(attempts):
            try:
                with stage(report, 'stash_http'):
                    response = self.api.post(
                        absolute_stash_url("graphql", self.url),
                        json={"query": query, "variables": variables or {}},
                        timeout=self.request_timeout, allow_redirects=False,
                    )
            except (RequestsConnectionError, RequestsTimeout):
                if attempt + 1 == attempts:
                    raise
                time.sleep(0.5 * 2 ** attempt)
                continue
            if report is not None:
                report.response_size(len(response.content))
            if response.status_code not in retry_statuses or attempt + 1 == attempts:
                break
            time.sleep(0.5 * 2 ** attempt)
        response.raise_for_status()
        if 300 <= response.status_code < 400:
            raise RuntimeError("Stash redirected GraphQL; configure STASH_URL to its final URL.")
        with stage(report, 'stash_json_decode'):
            payload = response.json()
        if payload.get("errors"):
            raise RuntimeError("GraphQL: " + "; ".join(
                error.get("message", "Unknown error") for error in payload["errors"]
            ))
        if not isinstance(payload.get("data"), dict):
            raise RuntimeError("Stash returned no GraphQL data.")
        return payload["data"]

    def get_media(self, url: str) -> bytes:
        # ApiKey is a custom header; requests does not remove it on redirects.
        # Follow same-origin redirects explicitly, never forwarding it elsewhere.
        for _ in range(6):
            if origin(url) != origin(self.url):
                raise ValueError("Media URL uses a different Stash origin; check STASH_URL.")
            response = self.media.get(url, timeout=self.request_timeout, allow_redirects=False)
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    raise RuntimeError("Media redirect has no Location header.")
                url = urljoin(url, location)
                continue
            response.raise_for_status()
            return response.content
        raise RuntimeError("Too many media redirects.")
