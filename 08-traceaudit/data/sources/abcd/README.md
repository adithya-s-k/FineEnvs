# ABCD: source data for trace auditing

Retained from the earlier dataset investigation. This is a data source for the TraceAudit reviewer,
not an environment for training an agent to perform customer-service actions.

Source: [ASAPP Research's Action-Based Conversations Dataset](https://github.com/asappresearch/abcd),
pinned to commit `6b8700ce67c6b37b062dd7a60abc76d7ef832a97`.
The dataset contains collected human-to-human conversations in fictional customer-service scenarios,
with scenario fields, policy guidelines, and annotated actions. These are not private Nurture Boss logs.

## Retained files

- `source.json`: original URLs, pinned revision, file sizes, and SHA-256 hashes.
- `LICENSE`: the original ASAPP Research MIT notice.
- `inventory.json`: counts and annotation-readiness notes from the retained archive.
- `raw/abcd_v1.1.json.gz`: the complete downloaded conversation archive, kept local and ignored by Git.
- `raw/kb.json` and `raw/guidelines.json`: source workflow definitions and policy context.
- `raw/source.json` and `raw/LICENSE`: the original download manifest and license.

To restore the ignored raw files, run from the repository root:

```bash
python 08-traceaudit/adapters/fetch_abcd.py
```

The downloader uses only the pinned files in `source.json` and verifies their sizes and hashes.

## How TraceAudit can use it

Use conversations as evidence to inspect and source policies as context when available and applicable.
Reviewers can investigate a claim/tool contradiction, an omitted required action, or an unresolved
request. Preserve the source conversation and action identifiers when building a normalized trace.

ABCD's action annotations and source agent behavior are not independent gold failure verdicts. Each
candidate needs an applicable requirement, sufficient evidence, and human confirmation before its
failure label enters the reward. A noncanonical action sequence is not automatically a failed run.

The complete source archive is retained. The former simulator's 14 selected phone-change scenarios,
42 generated tasks, synthetic retry/handoff variants, and scripted policies are not the TraceAudit
dataset. No auditor failure labels are assigned by retaining this source.
