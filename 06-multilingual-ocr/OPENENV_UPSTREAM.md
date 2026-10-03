# What this experiment could contribute to OpenEnv

The strongest first contribution is a small dataset-serving guide and reference implementation
using the existing TaskProvider API, backed by a full corpus and measured cache behavior. Then we can propose reusable helpers with evidence from
both Nayana and LaTeX OCR. No upstream issue or PR has been opened for these proposals yet.

## Existing foundation

OpenEnv already defines the optional `TaskProvider` protocol: split and task discovery,
counts, indexed lookup, and ranges. Discovery is metadata-only; reset selects the task.
The current HTTP implementation creates an environment from the factory for each discovery
request and closes it afterward. Our factory already shares one immutable catalog between
these short-lived environments, so a core change is not required to serve Nayana.

Verified against OpenEnv main at `a3ee76444a8d1ec93a255db079bc848fc0540ad9`:
[Task API guide](https://huggingface.co/docs/openenv/guides/task-api),
[TaskProvider interface](https://github.com/huggingface/OpenEnv/blob/a3ee76444a8d1ec93a255db079bc848fc0540ad9/src/openenv/core/env_server/interfaces.py),
[HTTP discovery dispatch](https://github.com/huggingface/OpenEnv/blob/a3ee76444a8d1ec93a255db079bc848fc0540ad9/src/openenv/core/env_server/http_server.py).
These are proposals, not claims that upstream lacks all related work.

## 1. Dataset catalog and resource lifecycle

Extract a small reference TaskProvider backed by an immutable full-corpus index: stable task IDs,
snapshot identity, metadata-only bounded ranges, and a shared read-only resource whose lifetime
is independent of one episode. The guide should show metadata-only indexing, lazy bucket-backed media loading, and stable-ID reset
with two simultaneous sessions. Keep Datasets and SQLite as optional example dependencies.

Start with documentation and the existing factory closure pattern. If multiple environments
need it, propose `create_app(..., task_provider=catalog)` as an optional, backward-compatible
way to serve discovery without constructing an expensive episode environment. Existing factory
dispatch remains the fallback; define ownership/close behavior explicitly and add lifecycle tests.

Nayana evidence: [corpus.py](envs/nayana_ocr/data/corpus.py),
[index.py](envs/nayana_ocr/data/index.py), [app.py](envs/nayana_ocr/server/app.py).
Do not standardize Nayana's language fields, document hash split, SQLite schema, or crop policy
in OpenEnv core. Split counts must describe the actual indexed tasks, including exclusions and lazy image-validation limits.

## 2. Binary media reference and client helper

Propose an optional typed media reference with a URL/path, MIME type, content SHA-256, and
dimensions/byte size where applicable. A client resolver could verify hashes and share a
byte-bounded cache across repeated rollouts. The same design should support image, audio,
and video environments; it should not require every payload to carry base64 data.

Before a core API proposal, demonstrate it in two environments and specify relative URL
resolution, authentication, redirects, failures, size limits, cache eviction, and compatibility
with existing inline observations. Never fetch arbitrary references without the caller's
URL/authentication policy. This experiment supplies image-only evidence, not a completed generic
media protocol.

The corpus implementation also supplies coalesced row-group prefetch, atomic cache publication,
reader leases, eviction, bounded lock metadata, source-identity checks, and reconstruction after
asset eviction. These belong in an optional storage/media helper, not an HF-specific dependency
of every environment.

Nayana evidence: [observation fields](envs/nayana_ocr/models.py),
[binary endpoint](envs/nayana_ocr/server/app.py), [AssetCache](envs/nayana_ocr/training.py).

## 3. Replay conformance tests

Provide opt-in test helpers that assert: the same task ID and snapshot produce identical input
across fresh sessions; episodes have independent state; discovery does not advance sampling;
reset replays an item; references stay out of the declared rollout API; and an unexpected
snapshot change fails loudly. Include zero-count splits and bounded range errors.

Keep the core test helper independent of TRL. Show TRL group repetition in an integration
example: repeat a selected task ID for every completion, then let each rollout own a session.
The reference-answer assertion applies to environments that declare hidden labels, not every
OpenEnv task. Nayana's separate Gradio playground intentionally reveals labels after scoring.

Nayana evidence: [service smoke](envs/nayana_ocr/smoke.py),
[environment tests](envs/nayana_ocr/tests), and [training adapter](envs/nayana_ocr/training.py).

## Suggested order and limits

1. Submit a guide/reference example for dataset preparation, TaskProvider, shared resources,
   and deterministic reset, with transport-level replay tests.
2. Validate media references/cache behavior in Nayana and LaTeX OCR, then propose the smallest
   common helper. Keep modality-specific transforms in environment packages.
3. Consider explicit TaskProvider injection if factory construction remains a measured cost.

Keep OCR rewards, Unicode normalization, geometric reading order, annotation masks, and the
document partition policy in HuggingEnvs. A general live stream scheduler with distributed
leases/eviction is a larger design: this experiment implements a complete immutable index, bounded local caches, and a block
iterator with a replayable consumer cursor. It does not implement cross-host distributed leases
or claim optimizer checkpoint replay. The source-group prefetch endpoint and cursor are useful
reference examples before proposing a generic optional scheduling protocol. Datasets iterator state also does not imply exact
shuffle-buffer or training-checkpoint replay. See [REPRODUCE.md](REPRODUCE.md) for those boundaries.
