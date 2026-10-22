# ATIF schema provenance

`envs/trace_audit/core/atif-v1.7.schema.json` was generated from Harbor's Pydantic `Trajectory` model
at commit `209a48a7424218828465c3c716b042f2b381dbed`, then modified to restrict the schema-version
enum/default to `ATIF-v1.7`. Harbor is licensed under **Apache-2.0**; its licence is retained in
`HARBOR_LICENSE`. This schema and cached upstream validation modules retain that upstream provenance.

`provenance.json` records the schema SHA-256 and exact hashes of the official model/validator files.
`tools/validate_harbor.py --fetch` restores only those pure validation modules, not an agent runtime.
Three local namespace stubs allow their imports without executing unrelated Harbor package setup.
All validation logic is unmodified upstream source. Version restrictions and v1 admission rules
(complete text traces, embedded references, normalized errors) are checked separately by our core.
