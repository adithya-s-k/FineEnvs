"""Restore capture credentials after HF Jobs authenticates the outer request."""
import hmac
from hf_endpoint_bridge import transport_key


class CaptureTransportAuth:
    def __init__(self, app, token):
        self.app = app
        self.key = transport_key(token).encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        forwarded = any(k.startswith(b"x-openenv-forwarded-") for k in headers)
        if forwarded:
            supplied = headers.get(b"x-openenv-transport-key", b"")
            if not hmac.compare_digest(supplied, self.key):
                await send({"type": "http.response.start", "status": 401, "headers": []})
                await send({"type": "http.response.body", "body": b"Invalid transport credentials"})
                return
            restored = [(k, v) for k, v in scope["headers"]
                        if k not in (b"authorization", b"x-api-key", b"x-goog-api-key", b"x-openenv-transport-key")
                        and not k.startswith(b"x-openenv-forwarded-")]
            for key in (b"authorization", b"x-api-key", b"x-goog-api-key"):
                value = headers.get(b"x-openenv-forwarded-" + key)
                if value:
                    restored.append((key, value))
            scope = {**scope, "headers": restored}
        await self.app(scope, receive, send)
