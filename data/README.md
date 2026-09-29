# Local data

Participant-level data are not distributed. Acquisition creates `raw/` and cohort construction creates `processed/`; their contents are ignored by Git. Aggregate provenance, cohort summaries, and the variable dictionary are committed under `metadata/`.

Run `python -m medintel acquire --data-dir data`, followed by `python -m medintel build-cohort --data-dir data` and `python -m medintel eda --data-dir data`. Acquisition refuses to overwrite a recorded snapshot unless `--overwrite` is supplied.

Keep downloaded XPT files unchanged. Record provenance and validation summaries separately from participant records. Consult `docs/data-protocol.md` before acquisition. Do not put patient uploads or credentials here.
