# Theory-discovery report

This is the minimal, review-ready package for the completed held-out run, built on the Prime orchestration architecture. It
contains the final results, frozen method contracts, audit metadata, and the
small deterministic execution receipts needed to inspect every conclusion.

## Headline result

Eight pre-frozen directional tests were run on 493 held-out records after 16
atomic constructs were measured by local Qwen3.8. Five results were directionally
consistent and three were directional contradictions.

| Prediction | Result |
| --- | --- |
| Intrusion co-occurrence | consistent |
| Worry / tiredness | contradiction |
| Inadequacy loop | contradiction |
| Monitoring drains | contradiction |
| Rest-time marker | consistent |
| Burden depletion | consistent |
| Meaning collapse | consistent |
| Anxiety attenuated | consistent |

Read `RESULT.md` before interpreting these labels. A positive directional
difference is sufficient for “consistent” in the frozen protocol; this package
does not claim causal identification, statistical significance, or proof of a
theory.

## Contents

- `methods/` — graph, theories, predictions, original falsification tests,
  frozen atomic questions, and frozen execution contracts.
- `results/` — the eight current held-out results plus one JSON result and one
  deterministic Python receipt per prediction.
- `audit/` — run status, readiness, 16 measurement-run hashes, and the declared
  held-out partition schema.
- `scripts/` — the measurement, deterministic execution, and export code used
  by the run. They are included for inspection; this package intentionally does
  not bundle the full runtime or model weights.
- `workflow/` — the complete canonical theory-discovery implementation, all
  discovery / measurement / compiler prompts, and the workflow test suite.
