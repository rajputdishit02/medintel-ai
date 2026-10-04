# Medical research assistant

The V0.7 assistant retrieves locally cached PubMed abstracts with TF-IDF and returns cited evidence. It can optionally use the OpenAI Responses API to synthesize a concise answer from the retrieved text. When no API key or API access is available, it returns cited extracts instead. Questions with no matching evidence are declined.

```bash
python -m medintel literature-acquire --data-dir data
python -m medintel literature-ask --data-dir data --offline --question "Why does calibration matter?"
python -m medintel literature-ask --data-dir data --question "Why does calibration matter?"
```

The acquisition step records the source, search terms, retrieval time, and corpus size in `data/metadata/literature-summary.json`. Abstracts remain local and are ignored by Git. Generated answers are never treated as clinical advice.

## Evaluation boundaries

- Retrieval tests use a fixed synthetic corpus and assert that a relevant article ranks first.
- Citation tests reject missing and out-of-range references.
- The assistant abstains when lexical similarity is below the documented threshold.
- Model output is accepted only when it contains valid source numbers; otherwise the extractive fallback is used.
- Retrieval relevance and valid citation syntax do not establish factual completeness or clinical validity.

To enable synthesis, store `OPENAI_API_KEY` in `.env.local`. Do not commit that file. API billing is separate from ChatGPT subscriptions; the extractive path is the supported no-cost fallback.
