"""HF transport credentials must not replace rollout or admin credentials."""
import unittest

from hf_endpoint_auth import CaptureTransportAuth
from hf_endpoint_bridge import transport_key
from openenv.core.harness.capture.sessions import SessionRegistry


class CaptureAuthTest(unittest.IsolatedAsyncioTestCase):
    async def check(self, headers, expected_session, expected_status=200):
        registry = SessionRegistry()
        registry.create("rollout-a")
        seen = []

        async def app(scope, receive, send):
            restored = {k.decode(): v.decode() for k, v in scope["headers"]}
            session = registry.resolve(restored, {})
            self.assertEqual(session.session_id if session else None, expected_session)
            self.assertNotIn("outer-token", str(restored))
            seen.append(200)

        async def send(message):
            if message["type"] == "http.response.start":
                seen.append(message["status"])

        middleware = CaptureTransportAuth(app, "outer-token")
        await middleware({"type": "http", "headers": headers}, None, send)
        self.assertEqual(seen, [expected_status])

    async def test_supported_credentials_preserve_session(self):
        for key in (b"authorization", b"x-api-key", b"x-goog-api-key"):
            with self.subTest(key=key):
                value = b"Bearer rollout-a" if key == b"authorization" else b"rollout-a"
                headers = [(b"authorization", b"Bearer outer-token"),
                           (b"x-openenv-forwarded-" + key, value),
                           (b"x-openenv-transport-key", transport_key("outer-token").encode())]
                await self.check(headers, "rollout-a")
                await self.check(headers[:-1], None, 401)

    async def test_local_credentials_unchanged(self):
        await self.check([(b"authorization", b"Bearer rollout-a")], "rollout-a")

    async def test_unknown_capture_session_rejected(self):
        await self.check([(b"authorization", b"Bearer unknown")], None)

    async def test_transport_does_not_grant_admin_identity(self):
        headers = [(b"authorization", b"Bearer outer-token"),
                   (b"x-openenv-forwarded-authorization", b"Bearer rollout-a"),
                   (b"x-openenv-transport-key", transport_key("outer-token").encode())]

        async def app(scope, receive, send):
            self.assertEqual(dict(scope["headers"])[b"authorization"], b"Bearer rollout-a")

        await CaptureTransportAuth(app, "outer-token")(
            {"type": "http", "path": "/sessions", "headers": headers}, None, None)


if __name__ == "__main__":
    unittest.main()
