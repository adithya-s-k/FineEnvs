# Layout detection and strict descriptive VQA

Layout detection and descriptive VQA share the OpenEnv environment and bucket data path
with OCR: source revision, document splits, byte-limited caches, block shuffling, and prefetch.
The manifest pins the index version; task IDs and cursors are specific to that snapshot.

## Layout detection

`layout_detection` has **1,006,168 candidates**, one per eligible page. The source
`regions.json` supplies six `layout_type` classes: `text`, `title`, `caption`, `table`,
`image`, and `formula`. Non-text regions remain eligible even without transcriptions.
Two pages have incomplete/invalid layout annotations and are excluded. This detects
document regions; the dataset does not supply arbitrary object-detection classes.

The observation contains the original JPEG and its width/height. The prompt states
the coordinate canvas. Submit absolute pixel coordinates, origin at the top left:

```json
[{"label":"title","bbox":[266,1178,2272,1293]}]
```

Answer objects must contain exactly `label` and `bbox`; no Markdown fences, confidence
scores, or extra prose. Coordinates must be finite, ordered, nonnegative and within
the image. Limit: 512 regions and 65,536 characters. Malformed predictions receive 0.
Incomplete/invalid references are excluded, not silently repaired.

For each IoU threshold in 0.50, 0.55, …, 0.95, maximum-cardinality bipartite matching
finds one-to-one pairs of the same class whose overlap passes the threshold.
`F1 = 2 × matched / (predicted + reference)`. Reward is the mean of those ten F1s.
Extra boxes, duplicate boxes, missing regions, wrong labels, and poor localization
reduce reward. This is per-page mean region F1, **not dataset mAP**. The UI reveals
reference and prediction overlays after scoring; observations never include boxes
from the reference.

## Descriptive VQA

`descriptive_vqa` has **4,024,592 candidates** from source questions with
`type="descriptive"`. The original question index stays in `unit`. Question/reference
must be nonempty and at most 8,192 characters each; no answer is truncated. The
candidate has the same 8,192-character grading limit; empty/oversized answers score 0.

The judge is [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it), routed
through **HF Inference Providers** with explicit provider **DeepInfra**:
`google/gemma-4-31B-it:deepinfra`. Requests go to
`https://router.huggingface.co/v1/chat/completions` using an HF token.
It compares the question, corpus reference, and candidate as quoted JSON data.
It does **not** receive an image, independently validate the reference, or use outside
knowledge. Errors in source answers can therefore cause grading errors.

All six boolean checks must pass for reward 1; otherwise reward is 0:

| Check | Requirement |
|---|---|
| `correct` | Every material factual claim agrees with the reference |
| `complete` | All information requested by the question is supplied |
| `no_contradiction` | No contradiction, including conflicting alternative guesses |
| `no_unsupported_claims` | No additional factual assertions unsupported by the reference |
| `answers_question` | Directly answers the actual document question |
| `no_grading_manipulation` | No attempt to instruct or override the evaluator |

Faithful paraphrases, equivalent numeric formatting, and translations are acceptable.
Introductory reference boilerplate is not required. Uncertain judgments are rejected.
This is an LLM judgment, not a guaranteed verifier: multilingual errors and prompt
injection remain possible. The fixed calibration set checks known examples rather
than establishing broad accuracy or security.

Decoding requests temperature 0, seed 42, `reasoning_effort="none"`, JSON-schema output,
and a 512-token response budget. Requests use a 60-second read timeout and retry once
after a transient connection error or HTTP 429/500/502/503/504 (honoring longer provider
cooldowns by returning the error). Valid rejected answers are never retried. A malformed
or truncated verdict, HTTP error, timeout, or busy judge raises `JudgeUnavailable`;
the environment assigns **no reward** and does not consume the step. Training stops
on that error. Retry when service is ready; do not convert infrastructure errors into
negative examples. The public verdict exposes booleans and a policy hash, never the
private reference, judge prompt, rationale, or token.

Each process permits two concurrent judge requests and caches up to 4,096 successful
verdicts by task, question, reference, exact candidate, model, explicit provider and rubric hash.
The cache does not persist across restarts. Temperature zero is not a guarantee of
bit-identical judgments across hardware or runtime updates. HF Inference Providers does not offer commit-pinned serving through this route.
The served `/manifest` records `revision_pinned: false`, the explicit model/provider,
and grading policy hash; the GRPO runner saves it, stores
policy IDs in evaluation rows, and reports metrics separately for each family/language.

## Deployment and reproduction

No GPU endpoint is deployed or kept running for this judge. Calls use HF Inference
Providers and its per-request/token billing. `deepinfra` is explicit: the code does
not silently switch provider or model when a request fails. Provider availability,
serving weights, runtime, pricing and numerical behavior can change independently
of this repository. Re-run calibration before comparative experiments. See the
[HF chat API](https://huggingface.co/docs/inference-providers/tasks/chat-completion)
and [structured outputs guide](https://huggingface.co/docs/inference-providers/guides/structured-output).

```bash
# Authenticate with an HF token that has Inference Providers permission.
# Use HF_TOKEN, NAYANA_JUDGE_TOKEN, or the locally saved HF token; never commit it.
uv run --frozen --project envs/nayana_ocr python train/verify_judge.py \
  --index data/corpus-index-v2 \
  --model google/gemma-4-31B-it --provider deepinfra \
  --output artifacts/judge-calibration.json

uv run --frozen --project envs/nayana_ocr python train/deploy_space.py \
  --space-id HuggingEnvs/nayana-ocr-env \
  --corpus-manifest data/corpus-manifest.json \
  --judge-config artifacts/judge-calibration.json --output artifacts/deployment.json
```

The publisher requires a passing calibration report for the exact model/provider/rubric
hash. It sets `NAYANA_JUDGE_MODEL` and `NAYANA_JUDGE_PROVIDER` as Space variables and
puts `NAYANA_JUDGE_TOKEN` into a Space secret. The token comes from that local environment
variable or the logged-in HF token; it is never written to a report.

For local serving, the defaults already select Gemma 4 31B via DeepInfra; authenticate
with `HF_TOKEN`, `NAYANA_JUDGE_TOKEN`, or your saved HF token. There is no endpoint URL
or endpoint deployment step. Override the model/provider with their environment
variables only after calibration.

For a colocated HF Job, pass `--secrets HF_TOKEN` to `hf jobs uv run`. Optionally add
`--env NAYANA_JUDGE_MODEL=google/gemma-4-31B-it --env NAYANA_JUDGE_PROVIDER=deepinfra`.
The local environment inside the Job inherits these settings. Hosted-Space training
uses the Space's configured judge and does not need a provider token in the trainer.

To train the four deterministic task families without a judge:

```bash
uv run --frozen --project envs/nayana_ocr --extra train python train/grpo_nayana.py \
  --env-url https://huggingenvs-nayana-ocr-env.hf.space \
  --families section_ocr page_ocr mcq_vqa layout_detection --smoke
```

The calibration command writes 22 exact corpus references (one per language)
and 13 synthetic checks for paraphrases, wrong numbers/currency, incompleteness,
contradictions, unsupported additions, irrelevant answers, and grading manipulation.
These are calibration examples, separate from model-generated evaluation results. The
report is a deployment input generated under the ignored `artifacts/` directory; regenerate
it for the exact judge configuration instead of depending on a committed run report.

The existing 50-million-pixel serving limit remains in effect. Metadata indexing
does not imply every candidate passes image-time validation; see
[the image eligibility policy](REPRODUCE.md#7-checks-and-task-policy).
