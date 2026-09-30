Independently review exactly one `review_claim`. Do not propose concepts,
relations, processes, memos, or new evidence. Use only supplied material; each
evidence item needs a supplied `record_id` and brief original `quote`.

For a relationship, `review_records` contains support, source-only,
target-only, and explicit-contradiction cohorts. Cohort labels are retrieval
aids, not conclusions. For a process edge, read every supplied corpus record:
plausible sequence, co-occurrence, or indirect A → B → C never establishes an
edge. With `review_batch`, judge only that batch. With `batch_reviews` and
`review_synthesis`, synthesize the supplied decisions/counts without inventing
or rereading unavailable evidence; an `intermediate` synthesis is provisional.

For a relationship claim, return one decision for its `claim_id`:

- `RETAIN` only when warranted.
- `NARROW` when wording or conditions must change; include
  `revised_relationship` and at least one condition.
- `REJECT` when unsupported; use `assessment: "unsupported"` and
  `status: "insufficient_evidence"`.

Use `explicitly_expressed` only for a directly stated arrow.
`repeated_comparison` and `tentative_theoretical_inference` need two distinct
record IDs. Source-only, target-only, contradictory, and indirect cases are
not extra support; cross-case patterns remain tentative unless directly stated.

For a process edge, return only `expected_output`. `supported` is directly
warranted; `conditional` needs named boundary conditions; `tentative` has
mixed/unresolved support; `rejected` is unsupported/contradicted. Every
non-rejected edge needs direct supporting evidence; retain explicit
counterexamples and relevant boundaries. A rejected edge rejects its process
as written.

Use concise plain English for phrases, conditions, and rationale. Keep the
rationale near 90 words and never above 120 words. Keep IDs, enum values,
JSON keys, and quotes exactly as required. Return JSON only, following
`expected_output`.
