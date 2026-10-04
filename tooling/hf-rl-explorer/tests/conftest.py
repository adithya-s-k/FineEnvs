"""One sandbox for every test module: storage and caches in a temp folder, a fixed session secret, no warm-up."""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="rlx-test-")
os.environ.setdefault("RLX_TEST_TMP", _TMP)
os.environ.update(STORAGE_DIR=os.environ["RLX_TEST_TMP"], RLX_CACHE_DIR=f"{os.environ['RLX_TEST_TMP']}/cache", OAUTH_CLIENT_ID="test-client",
                  SESSION_SECRET="test-secret", RLX_WARM="0", RLX_ADMIN_ORG="FineEnvs")
