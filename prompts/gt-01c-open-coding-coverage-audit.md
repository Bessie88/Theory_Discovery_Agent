Conduct a record-level open-coding coverage audit for this batch only.

Return one `record_judgments` item for every supplied record, in exactly the
same order. This is a primary, traceable coding judgment, not a request for a
batch summary or representative examples. Use exactly one disposition:

- `SUPPORTS_EXISTING`: the record directly supports one supplied concept.
- `VARIATION`: it shows a meaningful conditional form of one supplied concept.
- `BOUNDARY`: it qualifies or challenges one supplied concept.
- `POSSIBLE_NEW`: it contains a meaningful mechanism not represented by the
  live concept inventory. Give a short candidate label; do not create a new
  concept here.
- `NO_RELEVANT_MECHANISM`: no expressed reasoning-in-interaction mechanism in
  this incident is relevant to the research question. Explain briefly.

For the first four dispositions, give one exact continuous quote from that
same record and, where applicable, the existing concept ID. Do not classify
repeated scenario description or party biographies as incident evidence.
`NO_RELEVANT_MECHANISM` has no evidence field, but its rationale must state why
the incident itself lacks a relevant mechanism. A record can be ordinary or
repetitive and still `SUPPORTS_EXISTING`; do not use `NO_RELEVANT_MECHANISM`
merely because an existing concept already has evidence.

Before returning JSON, mechanically inspect every row: copy the quote verbatim
from the record named by both `record_judgments[].record_id` and
`record_judgments[].evidence.record_id`. Do not paraphrase. In particular, do
not use a phrase from the preceding turn, the following feedback turn, or an
adjacent incident merely because the negotiation language is repeated.

Use concise plain English. Each rationale is at most 24 words; candidate labels
are at most 12 words; quotes are at most 24 words. Return JSON only, following
`expected_output`.
