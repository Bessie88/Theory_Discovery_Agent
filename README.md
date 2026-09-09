# Theory-discovery report

This is the minimal, review-ready package for the completed held-out run, built on the Prime orchestration architecture. It
contains the final results and frozen method contracts.

```mermaid
flowchart TD
    A[Research question, thematic graph, evidence, and discovery documents] --> B[Import thematic graph]
    B --> C[Discover candidate relationships]
    C --> D[Generate evidence-grounded theories]
    D --> E[Generate falsifiable predictions]
    E --> F[Create atomic measurement specifications]
    F --> G[Compile frozen falsification specifications]
    G --> H{Missing atomic construct?}
    H -- Yes --> I[Repair measurement specifications]
    I --> G
    H -- No --> J[Freeze measurement and decision rules]
    J --> K[Measure held-out validation data]
    K --> L[Run deterministic falsification tests]
    L --> M[Validation records]
```
