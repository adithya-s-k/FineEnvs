# Verification results

These are pipeline checks, not model evaluations or evidence of reward improvement.

| Artifact | What it records |
|---|---|
| `preparation-real-data.json` | Pinned revision, configuration, counts, exclusion reasons, media footprint, startup time |
| `smoke-real-data.json` | Actual HTTP/WebSocket and training-adapter checks across 4 languages × 2 task families |
| `smoke-fixture.json` | Equivalent transport checks on synthetic fixtures |
| `smoke-docker.json` | Real-data smoke against the built Linux ARM64 Docker image |
| `gradio-check.json` | Playground controls across all eight language/task groups and separate user sessions |
| `verification.json` | Unit, sampler, notebook, and container verification summary |

The real-data run prepared two pages per language for `en`, `kn`, `hi`, and `ar`. It produced
37 tasks, stored 5,945,947 media bytes, and excluded three ambiguous MCQ answers. It deferred
32 descriptive VQA questions. All sampled documents landed in train; validation and test are
empty. This prefix is a service smoke fixture, not an evaluation dataset.

Each checked task received an empty answer and a known reference answer in separate sessions.
The expected rewards were 0 and 1. Two training adapter instances reset to the same ID and
shared one fetched image. References used for this oracle check were read from the local
catalog by the trusted test driver; public observations and discovery never return them.

The reported smoke elapsed time excludes dataset preparation and server startup. It is one
warm local observation and is not a p95 latency, bandwidth, RAM, or throughput benchmark.

No GPU optimizer step, held-out model comparison, public Space deployment, or HF Jobs run
has been recorded yet. Training checkpoint replay is not yet verified. Local and container
verification details are recorded separately as implementation checks.
