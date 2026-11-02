"""Official Mapillary Graph API client, throttling and bounded retries.

Tokens are sent ONLY in the Authorization header to graph.mapillary.com;
neither tokens nor signed thumbnail URLs are written to download metadata.
"""

from email.utils import parsedate_to_datetime
import json
from http.client import HTTPException
import random
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class DownloadError(RuntimeError):
    pass


class RetryableDownloadError(DownloadError):
    """A transient transport/response failure after bounded request retries."""

    pass


class MapillaryHTTPError(DownloadError):
    def __init__(self, status, host="graph.mapillary.com", *, query_too_large=False):
        self.status = status
        self.query_too_large = query_too_large
        self.retryable = status in (429, 500, 502, 503, 504)
        super().__init__(f"HTTP {status} from {host}")


class ThumbnailUnavailable(DownloadError):
    """The requested thumbnail variant is absent, rather than a network error."""

    pass


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        target = urlsplit(new_url)
        if target.scheme != "https":
            raise DownloadError("Refusing a redirect away from HTTPS")
        if request.get_header("Authorization") and target.netloc != urlsplit(request.full_url).netloc:
            raise DownloadError("Refusing an authenticated redirect outside the API host")
        return super().redirect_request(request, response, code, message, headers, new_url)


urlopen = build_opener(SafeRedirect()).open


class MapillaryClient:
    def __init__(self, token, requests_per_second=2.0, retries=5, timeout=45, on_retry=None):
        if not token or requests_per_second <= 0 or retries < 0:
            raise ValueError("token and a positive request rate are required")
        self.token, self.retries, self.timeout = token, retries, timeout
        self.interval, self.next_request = 1 / requests_per_second, 0.0
        self.lock = threading.Lock()
        self.on_retry = on_retry

    def _report_retry(self, attempt, wait, status=None):
        if self.on_retry:
            self.on_retry({"event": "request_retry", "attempt": attempt + 1,
                           "max_retries": self.retries, "wait_seconds": round(wait, 1), "status": status})

    def _throttle(self):
        with self.lock:
            wait = max(0, self.next_request - time.monotonic())
            self.next_request = max(self.next_request, time.monotonic()) + self.interval
        if wait:
            time.sleep(wait)

    def _read(self, url, *, authenticated=False, max_bytes=32 * 1024 * 1024):
        parts = urlsplit(url)
        if parts.scheme != "https":
            raise DownloadError("Only HTTPS downloads are supported")
        if authenticated and parts.netloc != "graph.mapillary.com":
            raise DownloadError("Refusing to send credentials outside Mapillary Graph API")
        headers = {"User-Agent": "SVGT-Training/1.0"}
        if authenticated:
            headers["Authorization"] = "OAuth " + self.token
        for attempt in range(self.retries + 1):
            self._throttle()
            try:
                with urlopen(Request(url, headers=headers), timeout=self.timeout) as response:
                    content = response.read(max_bytes + 1)
                    if len(content) > max_bytes:
                        raise DownloadError("Response exceeded the configured size limit")
                    return content
            except HTTPError as error:
                code = error.code
                retry_after = error.headers.get("Retry-After")
                # Classify known size errors, but never log the raw response,
                # request URL, signed thumbnail URLs or credentials.
                try:
                    body = (error.read(8192) or b"").lower()
                except (OSError, ValueError, AttributeError, TypeError, HTTPException):
                    body = b""
                too_large = any(fragment in body for fragment in (
                    b"bounding box area", b"bbox area", b"reduce the amount of data",
                    b"too much data", b"request area is too large",
                ))
                error.close()
                if too_large or code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    # Never include the URL, token, signed query or raw server body.
                    raise MapillaryHTTPError(code, parts.hostname, query_too_large=too_large) from None
                wait = min(60.0, 2 ** attempt + random.random())
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        try:
                            wait = max(wait, parsedate_to_datetime(retry_after).timestamp() - time.time())
                        except (ValueError, TypeError, OverflowError):
                            pass
                self._report_retry(attempt, wait, code)
                time.sleep(wait)
            except (URLError, TimeoutError, OSError, HTTPException):
                if attempt == self.retries:
                    raise RetryableDownloadError(f"Network request failed for {parts.hostname}; progress is retained") from None
                self._report_retry(attempt, min(60, 2 ** attempt))
                time.sleep(min(60, 2 ** attempt))

    def get(self, path, **params):
        url = "https://graph.mapillary.com/" + path + "?" + urlencode(params)
        try:
            result = json.loads(self._read(url, authenticated=True))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise RetryableDownloadError("Mapillary returned invalid JSON") from None
        if not isinstance(result, dict):
            raise RetryableDownloadError("Mapillary returned an unexpected response")
        if "error" in result:
            raise DownloadError("Mapillary API returned an error; check token permissions and request settings")
        return result

    def search(self, bbox, fields, *, after=None, limit=100):
        params = {"bbox": ",".join(str(value) for value in bbox), "fields": fields,
                  "is_pano": "false", "limit": limit}
        if after:
            params["after"] = after
        result = self.get("images", **params)
        paging = result.get("paging", {})
        cursor = None
        if paging.get("next"):
            cursor = paging.get("cursors", {}).get("after")
            if not cursor:
                cursor = parse_qs(urlsplit(paging["next"]).query).get("after", [None])[0]
            if not cursor:
                raise DownloadError("Pagination has a next page without an after cursor")
        return result.get("data", []), cursor

    def image_bytes(self, image_id, resolution):
        field = f"thumb_{resolution}_url"
        for refresh in range(2):
            metadata = self.get(str(image_id), fields=field)
            url = metadata.get(field)
            if not url:
                raise ThumbnailUnavailable(f"Image {image_id} has no requested thumbnail")
            try:
                return self._read(url)
            except DownloadError as error:
                if refresh or not any(code in str(error) for code in ("HTTP 403", "HTTP 404")):
                    raise
        raise DownloadError("Unable to refresh the thumbnail URL")
