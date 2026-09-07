# Exclusions

- `src/rule.py`: Searches beta and thresholds against HPA test labels.
- `src/poolval.py`: Validation-only feature pooling.
- `src/pos.py`: Validation-only position extraction.
- `src/importance.py`: Validation-only deduplication.
- `src/analyze.py`: Validation-only attribution analysis.
- `src/trainlast.py`: Empty file.
- `src/test.py`: Local CSV diagnostic, not model evaluation.

All validation records, validation embedding/index files, tuning outputs, checkpoints, and working-directory result folders are excluded. The damaged original test embedding archive and the partial recovered archive are not published as a complete test dataset.
