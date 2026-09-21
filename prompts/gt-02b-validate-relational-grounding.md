Independently review the one arrow in `review_claim`. Do not propose concepts,
relations, processes, memos, or new evidence. Use only the original text in
`review_records`. For every evidence item, return `record_id` and a brief `quote`.

For a relationship review, `review_records` contains every record named in all four
`evidence_sets`: support, source-without-target, target-without-source, and
explicit contradictions. Those cohort labels are mechanical retrieval, not
conclusions: decide from the wording whether they support, qualify, or
challenge this specific arrow.

For a process-edge review, `review_claim.requires_full_corpus_review` is true
and `review_records` contains the entire corpus. Read every supplied original
record. The same four evidence sets remain visible as an audit aid, but do not
limit the decision to them: identify support, source-only cases, target-only
cases, and explicit counterexamples across the full corpus. A process edge is
not licensed by a plausible sequence, co-occurrence, or an indirect chain.

If `review_batch` is present, this is one bounded part of an oversized review.
Read every record in its `review_records` list and return a provisional decision
for that part only; the complete cohort is distributed across all batches. Do
not treat records absent from this batch as support or counterevidence.

If `batch_reviews` and `review_synthesis` are present, this is the final
synthesis after every original record was independently reviewed in one batch.
There is no `review_records` text to reread. Use the batch reviews and the full
cohort counts to return the one final decision; do not invent new evidence.
When `review_synthesis.phase` is `intermediate`, return a provisional grouped
decision that faithfully carries forward the supplied batch reviews; it will be
reviewed again in a later synthesis step.

When `review_claim.claim_type` is `relationship`, use `assessment` as follows:
use `explicitly_expressed` when a single supporting record
directly states the arrow. Use `repeated_comparison` or
`tentative_theoretical_inference` only when `supporting_records` contains at
least two distinct record IDs. Do not count source-only, target-only, or
contradictory cohorts as additional supporting evidence. If one supporting
record does not directly state the arrow, reject it rather than treating those
other cohorts as new support.

For a relationship review, return one decision for this `claim_id`:

- `RETAIN` when the arrow is warranted;
- `NARROW` when weaker wording or explicit conditions are required (include a
  revised phrase for a relationship and at least one condition);
- `REJECT` when unsupported; then use `assessment: "unsupported"` and
  `status: "insufficient_evidence"`.

Use the other assessment values only when justified by the wording. Cross-case
patterns remain tentative unless a record states the link. Co-occurrence and
an indirect A → B → C path are not direct arrows.

For a process-edge review, do not return `decision`, `assessment`, or `status`.
Return the fields required by `expected_output`:

- `edge_status: "supported"` only for a directly warranted edge with no
  qualifying condition;
- `edge_status: "conditional"` when the arrow is warranted only under named
  boundary conditions; list every relevant condition in `boundary_conditions`;
- `edge_status: "tentative"` when there is some direct support but the full
  corpus leaves the edge unresolved or mixed;
- `edge_status: "rejected"` when the edge is unsupported or contradicted.

For every non-rejected process edge, include at least one direct supporting
item in `supporting_evidence`. Put explicit counterexamples in
`negative_evidence`, preserve all relevant boundary conditions, and explain the
decision in `rationale`. A rejected edge rejects its whole process as written.

Use plain English for revised phrases, conditions, and explanations; keep quoted
evidence, JSON keys, IDs, and enum values exactly as required.

Return JSON only, following `expected_output`.
