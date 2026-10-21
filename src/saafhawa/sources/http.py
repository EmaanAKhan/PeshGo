"""HTTP with timeouts, retries and a simple rate limit."""

from __future__ import annotations

import threading
import time

import requests

_last_call: dict[str, float] = {}
_lock = threading.Lock()


class FetchError(RuntimeError):
    pass


def get(url: str, *, params: dict | None = None, headers: dict | None = None, per_minute: int | None = None,
        retries: int = 4, timeout: int = 60, key: str = "default") -> requests.Response:
    delay = 2.0
    for attempt in range(retries + 1):
        if per_minute:
            # Reserve the next free slot under a lock, so parallel downloads
            # together stay under the source's rate limit.
            with _lock:
                slot = max(time.monotonic(), _last_call.get(key, 0) + 60.0 / per_minute)
                _last_call[key] = slot
            wait = slot - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            if attempt == retries:
                raise FetchError(f"{url}: {e}") from e
        else:
            if r.status_code == 200:
                return r
            if r.status_code in (401, 403):
                raise FetchError(f"{url}: HTTP {r.status_code} (check the API key in .env)")
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise FetchError(f"{url}: HTTP {r.status_code}: {r.text[:200]}")
            retry_after = r.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = max(delay, float(retry_after))
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise FetchError(url)
