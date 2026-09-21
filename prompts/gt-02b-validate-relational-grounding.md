Independently review exactly one arrow: `review_claim`. Do not propose concepts,
relations, processes, memos, or new evidence. Use only supplied material. Every
evidence item must contain a supplied `record_id` and a brief original `quote`.

## Evidence scope

- For a **relationship** claim, `review_records` covers the four retrieval
  cohorts: support, source-only, target-only, and explicit contradictions.
  Cohort labels are retrieval aids, not conclusions: judge the wording yourself.
- For a **process edge**, `requires_full_corpus_review` is true. Read every
  supplied corpus record; identify support, source-only cases, target-only
  cases, and explicit counterexamples across the corpus. Plausible sequence,
  co-occurrence, or an indirect A → B → C chain never establishes an edge.
- With `review_batch`, decide only the records in that batch; records absent
  from it are neither support nor counterevidence.
- With `batch_reviews` and `review_synthesis`, synthesize the supplied batch
  decisions and cohort counts. Do not invent evidence or reread unavailable
  text. An `intermediate` synthesis is provisional.

## Relationship claim

Return one decision for the supplied `claim_id`:

- `RETAIN` only when the arrow is warranted.
- `NARROW` when weaker wording or explicit conditions are needed; include
  `revised_relationship` and at least one condition.
- `REJECT` when unsupported; use `assessment: "unsupported"` and
  `status: "insufficient_evidence"`.

Use `explicitly_expressed` only when one record directly states the arrow.
`repeated_comparison` and `tentative_theoretical_inference` require support
from at least two distinct record IDs. Source-only, target-only, contradictory,
and indirect cases are not extra support. Cross-case patterns stay tentative
unless a record directly states the link.

## Process-edge claim

Return only the fields in `expected_output`, not relationship `decision`,
`assessment`, or `status`.

- `supported`: directly warranted without qualifying conditions.
- `conditional`: warranted only under named `boundary_conditions`.
- `tentative`: some direct support, but mixed or unresolved across the corpus.
- `rejected`: unsupported or contradicted.

Every non-rejected edge needs direct `supporting_evidence`; record explicit
counterexamples in `negative_evidence`, preserve relevant boundary conditions,
and explain the judgement in `rationale`. A rejected edge rejects its process
as written.

Use plain English for phrases, conditions, and rationale. Keep IDs, enum
values, JSON keys, and quotes exactly as required. Return JSON only, following
`expected_output`.

