Conduct Corbin & Strauss theoretical integration. Build an explanatory account,
not a theme summary, from supplied concepts, relations, processes, memos, and
negative cases.

`local_theory_memos` are fixed-input neighborhood analyses, not new categories.
Use their source IDs and `cross_neighborhood_relations` without merging
concepts or assuming causality. When `evidence_on_demand` is supplied, prefer
that original material over a conflicting or weak summary. Retain meaningful
status qualifiers, negative cases, and tentative alternatives.

Choose a core category only when it is well grounded, connects important parts
of the analysis, and explains variation. Otherwise return
`no_adequately_grounded_core_category` with `core_category_id: null`. Do not
invent a core category or claim saturation.

Separate well-grounded, tentative, and unresolved propositions. Each needs an
existing process/relation ID or evidence with `record_id` and a brief `quote`.
Preserve contradictions, gaps, alternatives, and uncertainty; do not strengthen
causal claims beyond the records. For an underdeveloped item, add a
`theoretical_sampling_need` describing the target, gap, needed evidence, and
current evidence.

Use concise plain English for the account, propositions, pathways, questions,
and sampling needs. Keep quotes, JSON keys, IDs, and enum values exactly as
required. Return JSON only, following `expected_output`.
