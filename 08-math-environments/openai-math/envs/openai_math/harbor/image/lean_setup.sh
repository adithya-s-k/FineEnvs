#!/usr/bin/env bash
# Builds the image shared by every openai/math task, for both the agent and the separate verifier:
#   - Lean v4.34.1 and Mathlib d13f23b, the versions openai/math pins (Mathlib only, no OAI library)
#   - Comparator, lean4export and landrun, used only by the verifier's grade.py
#   - every challenge statement, at /opt/openai-math/challenges/<Name>.lean
#   - /workspace, a Lake project where the agent writes Submission.lean
# All revisions are pinned. The openai/math library (the reference proofs) is never installed.
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

LEAN_TOOLCHAIN=leanprover/lean4:v4.34.1
OPENAI_MATH_REV=adc7f1241b42e322a6451854ab7e4b4c146bf78a
COMPARATOR_REV=d03acab154d269c06e60e4de7e4cc85deebff94b   # "bump toolchain to v4.34.0"; built with v4.34.1
LANDRUN_REV=811cfff51ceaf3d9843708aa6d22e9b84ccac8b4
GO_VERSION=1.24.7

apt-get update -qq
apt-get install -y -qq --no-install-recommends git curl ca-certificates build-essential xz-utils \
  zstd python3 procps iproute2 iptables tmux >/dev/null
rm -rf /var/lib/apt/lists/*
git config --system --add safe.directory '*'

export ELAN_HOME=/opt/elan PATH=/opt/elan/bin:$PATH
curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
  | sh -s -- -y --default-toolchain "$LEAN_TOOLCHAIN" --no-modify-path

# --- Verifier tools, built before Mathlib so the build trees are gone before the big download ---
curl -sSfL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" | tar -C /tmp -xz
GOBIN=/usr/local/bin GOPATH=/tmp/gopath GOCACHE=/tmp/gocache \
  /tmp/go/bin/go install "github.com/zouuup/landrun/cmd/landrun@${LANDRUN_REV}"
rm -rf /tmp/go /tmp/gopath /tmp/gocache

git clone -q https://github.com/leanprover/comparator.git /tmp/comparator
(
  cd /tmp/comparator
  git checkout -q "$COMPARATOR_REV"
  # Comparator passes only PATH/HOME/LEAN_ABORT_ON_PANIC into its landrun sandbox. Also pass
  # LEAN_NUM_THREADS: Lean sees every host CPU (64) rather than the sandbox's 4, so without it Lake
  # starts one Mathlib-loaded `lean` per module and an 8 GB sandbox runs out of memory. It bounds
  # parallelism only and does not affect what is checked.
  sed -i 's/envPass := #\["PATH", "HOME", "LEAN_ABORT_ON_PANIC"\]/envPass := #["PATH", "HOME", "LEAN_ABORT_ON_PANIC", "LEAN_NUM_THREADS"]/; s/envPass := #\["PATH", "HOME", "LEAN_PATH", "LEAN_ABORT_ON_PANIC"\]/envPass := #["PATH", "HOME", "LEAN_PATH", "LEAN_ABORT_ON_PANIC", "LEAN_NUM_THREADS"]/' Main.lean
  test "$(grep -c '"LEAN_NUM_THREADS"' Main.lean)" -eq 2
  echo "$LEAN_TOOLCHAIN" > lean-toolchain
  lake build lean4export comparator
  install -m 0755 .lake/build/bin/comparator /usr/local/bin/comparator
  install -m 0755 .lake/packages/lean4export/.lake/build/bin/lean4export /usr/local/bin/lean4export
)
rm -rf /tmp/comparator

# --- Lean project with Mathlib ---
# Packages are shallow clones at the manifest revisions, so Lake sees the right HEAD and never fetches.
mkdir -p /opt/lean-env/.lake/packages
cd /opt/lean-env
echo "$LEAN_TOOLCHAIN" > lean-toolchain
cp /opt/build/lakefile.lean /opt/build/lake-manifest.json .
python3 - <<'PY'
import json, subprocess
for p in json.load(open("lake-manifest.json"))["packages"]:
    d = f".lake/packages/{p['name']}"
    subprocess.run(["git", "init", "-q", d], check=True)
    subprocess.run(["git", "-C", d, "remote", "add", "origin", p["url"]], check=True)
    subprocess.run(["git", "-C", d, "fetch", "-q", "--depth", "1", "origin", p["rev"]], check=True)
    subprocess.run(["git", "-C", d, "checkout", "-q", "FETCH_HEAD"], check=True)
PY
# Linking native executables (Comparator above, Mathlib's cache tool here) is the only use of the
# static libraries, LLVM and gcc; `lean` and `lake build` of a library need none of them. Dropping
# them before the Mathlib download keeps the build, and the image, under Daytona's 10 GB disk.
lake exe cache --help >/dev/null 2>&1 || true
find /opt/elan/toolchains/*/lib -name '*.a' -delete
rm -f /opt/elan/toolchains/*/lib/libLLVM* /opt/elan/toolchains/*/lib/libclang-cpp*
apt-get purge -y -qq $(dpkg-query -W -f '${binary:Package}\n' build-essential 'gcc-13*' 'g++-13*' \
  'cpp-13*' 'libstdc++-13-dev*' 'libgcc-13-dev*' 2>/dev/null) >/dev/null || true
apt-get autoremove -y -qq >/dev/null || true
bash /opt/build/fetch_mathlib.sh
rm -f /opt/lean-env/.lake/packages/mathlib/.lake/build/bin/cache

# --- Challenge statements (public; the reference proofs are not fetched) ---
git init -q /tmp/oai && cd /tmp/oai
git remote add origin https://github.com/openai/math.git
git config core.sparseCheckout true
echo "lean/ComparatorChallenges/" > .git/info/sparse-checkout
git fetch -q --depth 1 --filter=blob:none origin "$OPENAI_MATH_REV"
git checkout -q FETCH_HEAD
mkdir -p /opt/openai-math/challenges
cp lean/ComparatorChallenges/*.lean /opt/openai-math/challenges/
cd / && rm -rf /tmp/oai
install -m 0755 /opt/build/pack_submission.py /opt/openai-math/pack_submission.py

# --- Agent workspace and the verifier's unprivileged user ---
mkdir -p /workspace/.lake
cp /opt/lean-env/lakefile.lean /opt/lean-env/lake-manifest.json /opt/lean-env/lean-toolchain /workspace/
ln -s /opt/lean-env/.lake/packages /workspace/.lake/packages
useradd --uid 2000 --create-home --shell /usr/sbin/nologin grader
mkdir -p /var/lib/oai-grade && chmod 0711 /var/lib/oai-grade

# Smoke test. The first build may rebuild a few dependency modules the cache lacks (they persist in
# the image); a second build must compile nothing but Submission, or agents would pay for it.
cd /workspace
printf 'import Mathlib\n\nexample : (2 : ℕ) + 2 = 4 := by norm_num\n' > Submission.lean
lake build Submission 2>&1 | grep -E 'Built|error' | grep -v 'Built Submission' | head -20 || true
echo '-- second build' >> Submission.lean
lake build Submission 2>&1 | tee /tmp/smoke.log
if grep 'Built' /tmp/smoke.log | grep -qv 'Built Submission'; then
  echo "the prebuilt Mathlib is incomplete: a second build still compiled dependencies" >&2; exit 1
fi
rm -rf Submission.lean .lake/build /tmp/smoke.log

# Mathlib's cache tool unpacks files as 0600. The verifier builds as the unprivileged `grader` user,
# so everything must be world-readable (and stays root-owned, so not writable).
chmod -R a+rX /opt/elan /opt/lean-env /opt/openai-math
# Same smoke test as `grader`, in a scratch project like the one grade.py lays out.
smoke=$(mktemp -d /var/lib/oai-grade/smoke.XXXX)
cp /opt/lean-env/lakefile.lean /opt/lean-env/lake-manifest.json /opt/lean-env/lean-toolchain "$smoke/"
mkdir "$smoke/.lake" && ln -s /opt/lean-env/.lake/packages "$smoke/.lake/packages"
printf 'import Mathlib\n\nexample : (2 : ℕ) + 2 = 4 := by norm_num\n' > "$smoke/Submission.lean"
chmod 0755 "$smoke" && chown grader "$smoke/.lake"
(cd "$smoke" && runuser -u grader -- env -i PATH="$(echo /opt/elan/toolchains/*/bin):/usr/bin:/bin" HOME="$smoke" \
  lake build Submission) 2>&1 | tee /tmp/smoke.log
if grep 'Built' /tmp/smoke.log | grep -qv 'Built Submission' || ! grep -q 'Build completed successfully' /tmp/smoke.log; then
  echo "the grader user cannot build against the prebuilt Mathlib" >&2; exit 1
fi
rm -rf "$smoke" /tmp/smoke.log

rm -rf /opt/build /root/.cache
df -h / | tail -1
du -sh /opt/elan /opt/lean-env
