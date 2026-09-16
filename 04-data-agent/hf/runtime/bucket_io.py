"""Bounded, idempotent retries for transient HF Bucket transfer failures."""
import json
import time


def sync_with_retry(api, source, destination, **options):
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(3):
        try:
            return api.sync_bucket(str(source), destination, quiet=True, **options)
        except (TimeoutError, ConnectionError, httpx.TransportError, HfHubHTTPError) as exc:
            if isinstance(exc, HfHubHTTPError):
                code = getattr(exc.response, "status_code", None)
                if code not in {429, 500, 502, 503, 504}:
                    raise
            if attempt == 2:
                raise
            # Xet can commit data before its response times out. Sync compares
            # existing content, so repeating the same transfer is safe.
            print(json.dumps({"bucket_retry": attempt + 1, "error_type": type(exc).__name__,
                              "destination": destination}), flush=True)
            time.sleep(30 * (attempt + 1))
