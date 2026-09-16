Independently review the one arrow in `review_claim`. Do not propose concepts,
relations, processes, memos, or new evidence. Use only the original text in
`review_records`.

Read every record named in all four `evidence_sets`: support,
source-without-target, target-without-source, and explicit contradictions.
Those cohort labels are mechanical retrieval, not conclusions: decide from the
wording whether they support, qualify, or challenge this specific arrow.

Return one decision for this `claim_id`:

- `RETAIN` when the arrow is warranted;
- `NARROW` when weaker wording or explicit conditions are required (include a
  revised phrase for a relationship and at least one condition);
- `REJECT` when unsupported; then use `assessment: "unsupported"` and
  `status: "insufficient_evidence"`.

Use the other assessment values only when justified by the wording. Cross-case
patterns remain tentative unless a record states the link. Co-occurrence and
an indirect A → B → C path are not direct arrows. A rejected process edge
rejects its process as written. Return JSON only, following `expected_output`.
