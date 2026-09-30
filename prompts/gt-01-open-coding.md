Conduct Corbin & Strauss open coding for this batch only. Code meaningful
segments, not every sentence; assign multiple codes only to distinct phenomena.
Mark every supplied record processed and return exactly one ordered
`record_judgments` item for it. A judgment is `SUPPORTS_EXISTING`, `VARIATION`,
`BOUNDARY`, `POSSIBLE_NEW`, or `NO_RELEVANT_MECHANISM`; it is a durable primary
coding decision, not a representative example. The first four require one
exact same-record quote. Existing-concept judgments require its ID;
`POSSIBLE_NEW` requires a short candidate label; `NO_RELEVANT_MECHANISM`
requires a brief reason and no evidence. Do not use `NO_RELEVANT_MECHANISM`
merely because an existing concept already has evidence.

Before returning JSON, mechanically check each judgment row: its
`record_id` and `evidence.record_id` must be identical, and the quote must be
copied only from that same full record. Adjacent negotiation turns often repeat
language; never borrow a quote or record ID from the preceding or following
turn.

Compare each update with the live inventory:

- `SAME`: reuse the concept and add evidence.
- `VARIATION`: retain an evidence-grounded conditional form.
- `NEW`: use only for a genuinely distinct, corpus-specific phenomenon.
- `CONTRADICTION`: retain boundary or opposing evidence.

With `indexed_retrieval`, `global_concept_index` is the complete comparison
inventory; `concept_inventory` supplies detailed retrieved items. Check every
possible `NEW` against the complete index, not only detailed items. Similarity
alone does not prove equivalence; use the appropriate existing decision unless
the definition and evidence justify a distinct concept.

Use `revised_definition` only for a material evidence-grounded refinement. Do
not create superficial wording variants, import external knowledge, or infer
relationships. Every evidence item needs a supplied `record_id` and brief
`quote`. Write a memo only for a material analytic change, variation,
contradiction, or unresolved question.

Use concise plain English for labels, definitions, variations, and memos. Keep
quoted evidence, JSON keys, IDs, and enum values exactly as required. Return
JSON only, following `expected_output`.
