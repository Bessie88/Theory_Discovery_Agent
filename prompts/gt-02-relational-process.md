Conduct Corbin & Strauss relational and process analysis for this batch against
the live inventories. Analyze conditions, actions/interactions, consequences,
and pathways only when the records support them; not every concept needs a
relationship.

Never treat co-occurrence, an indirect A → B → C sequence, or unrelated cases
as a direct arrow. Every relationship needs a supplied `record_id`, brief
`quote`, comparative basis, and `grounding_explanation`:

- `explicitly_expressed`: one record directly states the link.
- `repeated_comparison`: at least two records support the same link.
- `tentative_theoretical_inference`: clearly tentative cross-record inference
  with at least two records.

Use `SAME`, `VARIATION`, `NEW`, or `CONTRADICTION` against the relation/process
inventory. Preserve conditions and negative cases. Stage 2 may make explicit,
evidence-grounded concept/category updates only: refine a definition, revise a
supported placement, add a genuinely new category, or retain a variation or
boundary case; never silently reinterpret Stage 1.

Use natural-language relation phrases. Each process edge must name source,
relationship, target, and available candidate evidence. Cite an existing
same-direction relation ID when available; otherwise retain the supported
candidate edge for its independent full-corpus review. A process description or
plausible sequence is not arrow evidence. Write memos only for material
analytic developments.

Use concise plain English for all analytic text. Keep quotes, JSON keys, IDs,
and enum values exactly as required. Return JSON only, following
`expected_output`.
