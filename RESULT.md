# Result — school-burnout-psychological-distress-heldout-qwen38

Study question: "How may academic anxiety, burnout, and self-doubt interact in students' experiences of school burnout?"

**Status: `complete`.** Job `58655327` completed the frozen held-out protocol
on 2026-09-07: 16 frozen constructs were measured over all 493 held-out records
(7,888 labels), and eight frozen contracts were executed deterministically.

## Held-out results

| Prediction | Result | Frozen group-A vs group-B outcome prevalence |
| --- | --- | --- |
| `pred_intrusion_cooccur` | prediction consistent | 82.35% vs 29.68% (+52.67 pp) |
| `pred_worry_tiredness` | directional contradiction | 16.00% vs 51.90% (−35.90 pp) |
| `pred_inadequacy_loop` | directional contradiction | 0.00% vs 0.00% (0.00 pp) |
| `pred_monitoring_drains` | directional contradiction | 8.86% vs 60.84% (−51.98 pp) |
| `pred_resttime_marker` | prediction consistent | 24.85% vs 23.77% (+1.09 pp) |
| `pred_burden_depletion` | prediction consistent | 82.35% vs 40.00% (+42.35 pp) |
| `pred_meaning_collapse` | prediction consistent | 34.86% vs 0.00% (+34.86 pp) |
| `pred_anxiety_attenuated` | prediction consistent | 40.00% vs 6.67% (+33.33 pp) |

These are directional frozen-contract outcomes, not causal estimates,
significance tests, or proof of a theory. Qwen3.8 did not read held-out text
during discovery or test design; Qwen3.6 received only one frozen question and
one held-out text per independent request.

## Saved objects (from project/state.json)
- Thematic graph: 4 nodes, 3 `contains_theme` edges.
- Candidate relationships: 6 (association, sequence, mechanism, moderation among the three specific themes).
- Theories: 4 (2 general, 2 specific).
- Falsifiable predictions: 8 (one per theory law).
- Falsification tests: 8 (one per prediction), all `implementability: ready`, `relevance: direct`.

## Theories (proposals)
- **intrusion_depletion_theory** (general) — Intrusive anxiety depletes capacity (general). Laws: intrusion_cooccurrence, worry_tiredness_link.
- **selfappraisal_drain_theory** (general) — Self-appraisal drain (general). Laws: inadequacy_monitoring, monitoring_drains.
- **resttime_intrusion_theory** (specific) — Rest-time intrusion as the primary distress channel (specific). Laws: resttime_marker, burden_depletion.
- **meaning_collapse_theory** (specific) — Meaning collapse precedes automatic engagement (specific). Laws: meaning_collapse_first, anxiety_attenuated.

## Predictions (proposals)
- **pred_intrusion_cooccur** (theory `intrusion_depletion_theory`, law `intrusion_cooccurrence`)
- **pred_worry_tiredness** (theory `intrusion_depletion_theory`, law `worry_tiredness_link`)
- **pred_inadequacy_loop** (theory `selfappraisal_drain_theory`, law `inadequacy_monitoring`)
- **pred_monitoring_drains** (theory `selfappraisal_drain_theory`, law `monitoring_drains`)
- **pred_resttime_marker** (theory `resttime_intrusion_theory`, law `resttime_marker`)
- **pred_burden_depletion** (theory `resttime_intrusion_theory`, law `burden_depletion`)
- **pred_meaning_collapse** (theory `meaning_collapse_theory`, law `meaning_collapse_first`)
- **pred_anxiety_attenuated** (theory `meaning_collapse_theory`, law `anxiety_attenuated`)

## Falsification tests (proposals)
Each test is implementable using the supplied held-out fields only (`record_id`, `text_review`). Each test specifies a measurable null and alternative hypothesis, a coding rule applied to `text_review`, and the comparison to compute.
- **test_intrusion_cooccur** for `pred_intrusion_cooccur` — ready, relevance direct; required data: record_id, text_review
- **test_worry_tiredness** for `pred_worry_tiredness` — ready, relevance direct; required data: record_id, text_review
- **test_inadequacy_loop** for `pred_inadequacy_loop` — ready, relevance direct; required data: record_id, text_review
- **test_monitoring_drains** for `pred_monitoring_drains` — ready, relevance direct; required data: record_id, text_review
- **test_resttime_marker** for `pred_resttime_marker` — ready, relevance direct; required data: record_id, text_review
- **test_burden_depletion** for `pred_burden_depletion` — ready, relevance direct; required data: record_id, text_review
- **test_meaning_collapse** for `pred_meaning_collapse` — ready, relevance direct; required data: record_id, text_review
- **test_anxiety_attenuated** for `pred_anxiety_attenuated` — ready, relevance direct; required data: record_id, text_review

## Integrity checks
- Each of the 16 measurement artifacts has a matching frozen-contract hash,
  the same validation-input hash, and exactly 493 unique record IDs.
- All 7,888 labels parsed as valid `PRESENT` or `ABSENT`; none was silently
  converted from a malformed response, and there were no `UNCERTAIN` labels.
- Recomputing all eight execution receipts from the hash-checked labels exactly
  reproduces their stored statuses, summaries, and group counts.
- The 6 pre-v10 lexical pilot results remain only as legacy audit data in
  `project/state.json`; they do not count toward this result.

## Review files
- `stage-results/05a_frozen_measurement_specifications.json` — 16 frozen atomic questions.
- `stage-results/05_falsification_specifications.json` — frozen groups and outcomes.
- `stage-results/06_frozen_measurement_runs.json` — hashes for all 16 runs.
- `stage-results/07_validation_records.json` — only the eight current held-out results.
- `validation-execution/<prediction-id>.py` — deterministic receipt and group counts.
