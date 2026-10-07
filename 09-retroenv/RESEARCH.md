# Research and design notes

Research checked on 2026-09-22. Primary sources are linked so dataset and
verifier claims can be audited.

## Decisions grounded in the sources

- RDKit sanitization checks representability and allowed valences, and reaction
  SMARTS can be executed deterministically. This supports structure and template
  gates, not experimental feasibility. The distinction follows the
  [RDKit Book](https://www.rdkit.org/docs/RDKit_Book.html#chemical-reaction-handling)
  and its [sanitization/valence documentation](https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization).
- Bemis–Murcko scaffolds are computed using RDKit's documented
  [`MurckoScaffold`](https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html)
  implementation. Acyclic targets use their canonical product as the grouping
  key because an empty scaffold would collapse every acyclic compound into one
  split.
- ORD is the safest first public corpus: its official repository stores data as
  Parquet with serialized protobuf reactions and explicitly licenses the data
  CC-BY-SA-4.0 and code Apache-2.0. See the
  [official `ord-data` repository](https://github.com/open-reaction-database/ord-data).
- PaRoutes is useful for route and stock evaluation: its official repository
  provides two 10,000-route sets, two stock sets, and top-N route recovery
  tooling. The software repository is Apache-2.0, while the official v2 Zenodo
  record independently marks the benchmark artifacts CC-BY-4.0. They are kept
  as separate intake entries. See [MolecularAI/PaRoutes](https://github.com/MolecularAI/PaRoutes)
  and [PaRoutes v2 on Zenodo](https://zenodo.org/records/7341155).
- OpenEnv documents MCP as the agent boundary and the Gym-like reset/step API as
  the infrastructure boundary. RetroEnv exposes both, following the
  [official MCP lifecycle](https://github.com/huggingface/openenv/blob/main/docs/source/guides/mcp-environment-lifecycle.md).

## Corpus intake policy

The source list in the proposal is a discovery list, not an allowlist. A source
enters a release only after all of these are recorded:

1. exact upstream artifact and immutable version;
2. data license (not merely the code-repository license);
3. observed/generated/simulated evidence kind;
4. extraction and atom-mapping method;
5. redistribution and attribution obligations;
6. overlap audit against existing sources.

ORD, Lowe USPTO, CRD, CREED/CREED-CCV, and FREA now have pinned intake entries;
CREED remains isolated as non-commercial. PaRoutes v2 benchmark artifacts and
USPTO-LLM are tracked under their respective Zenodo CC-BY-4.0 records, while
SynRXN remains a per-artifact input. RXNGraphormer does not publish its stated
13M-reaction pretraining artifact in the official repository. AbSynth routes
require institutional-email access and carry a no-derivatives license, so they
cannot enter a transformable HF release. The pipeline will not invent a license
or erase evidence labels.

## Verifier boundary

The following are deterministic claims:

- SMILES parses and sanitizes under the pinned RDKit build;
- mapped atoms preserve element/isotope identity where mappings exist;
- every product's elemental atom inventory is contained in its proposed
  precursors even when the source record is unmapped;
- a hidden trusted template generates the canonical product from the proposed
  precursors;
- a proposal matches a hidden observed reaction record;
- the submitted route is a connected acyclic graph ending in the selected
  stock set.

The following are explicitly not claimed:

- that a reaction has useful yield or selectivity;
- that listed conditions are complete or compatible across the route;
- that a template match proves kinetics, mechanism, scale-up, safety, or
  laboratory feasibility;
- that a stock identifier guarantees current commercial availability or price.

Results therefore use `dataset_supported` and `template_supported`, never
`experimentally_validated`, unless a future source carries explicit experimental
evidence and the evaluation reports it separately.

## Recommended release order

1. Normalize 10,000 observed, license-audited single-step records.
2. Freeze global stock and scaffold-OOD splits before model experiments.
3. Run the leakage, duplicate, reward, and replay checks in this project.
4. Publish prompt-only development tasks and keep evaluation references server
   side.
5. Add provenance-backed 2–3 step routes and report them separately from
   single-step prediction.
6. Add template/search/open-model baselines only after the data split is frozen.
