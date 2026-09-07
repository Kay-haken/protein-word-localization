# Attribution-guided protein localization

Research code, training/test data, and an aggregated protein-word resource. Manuscripts and author identity information are excluded.

## Included material

- `src/`: training, representation, attribution, and fixed-threshold testing source snapshots. `src/rule.py`, which searches parameters on test labels, is excluded.
- `data/train/SEQ_total_train.csv`: 557,522 five-residue word records from 19,489 proteins.
- `data/train/training_proteins_and_labels.csv`: sequences and labels for exactly those 19,489 training-word IDs, derived from the supplied DeepLoc development table. This is not the complete 28,303-record development collection.
- `data/test/hpa_testset_5523.csv`: the complete supplied 5,523-record HPA label table. The source sequence for ENSP00000440575 is missing; 5,522 available sequences are included separately without truncation.
- `library/11_class_protein_words.docx` and the XLSX: readable 55-word display subset.
- `library/full_fragment_frequency_11_classes.csv.gz`: 665,788 class-word aggregate records; training/validation summary statistics are retained as library provenance. This is an aggregated resource, not a claim that all rows are final calibrated rules.

Validation datasets and test-set parameter-search files/results are intentionally excluded. Existing validation interfaces in training code remain because they are part of model training, but the private validation inputs are not distributed.

## Code change on 2026-09-07

`src/pool1.py` now sets `ig_weight = max(0.40 - 0.10 * cycle, 0.0)` and `raw_weight = 1.0 - ig_weight`. Thus cycle 4 onward uses Raw-only importance. This change has been checked for cycles 0–7; it does not rerun historical experiments or establish that reported metrics used this schedule.

## Running and reproducibility limits

Install the packages listed in `requirements.txt` in a suitable Python environment. Versions are not claimed to reconstruct the historical training environment. Most legacy scripts use paths from the original compute server; replace them with your own inputs before execution. Parameterized entry points include `src/step.py`, `src/train1new.py`, `src/pool1.py`, and `src/igfull.py`. Run scripts from `src/` because their local imports and subprocess paths use that directory.

Training requires externally prepared ESM/Protein Wordwise feature archives; existing validation-based training also requires separately supplied validation inputs. Historical model checkpoints and complete feature archives are not included. Syntax was checked for included Python files; end-to-end training and inference were not rerun. `logic.py` and `logic2.py` are legacy HPA inference scripts with fixed 0.5 threshold and hard-coded paths, not an assertion that they reproduce the manuscript's calibrated endpoint.

The source `step.py` uses an 80/20 random split; the presence of original `Partition` metadata in the source table does not establish homology-separated training/validation in this snapshot. Exact historical split/configuration provenance remains to be reconciled with the manuscript. Removing test-tuning scripts from the release does not establish that historical reported results were free from test-set selection.

The original HPA feature archive in the working directory is damaged; a recovered archive contains only 4,477 records. Neither is represented here as a complete 5,523-protein feature archive. Reported paired metrics use 5,162 records, a separate availability level.

## Data provenance

Development/HPA source: DeepLoc 2.0, Thumuluri et al., Nucleic Acids Research (2022), https://doi.org/10.1093/nar/gkac278 . The published paper's HPA comparison table evaluates 1,717 proteins; the local 5,523-record label collection should not be conflated with that published subset. Upstream data and code retain their original terms. No new license for third-party material is asserted.

`manifest.json` records source descriptions, exact file sizes, and SHA-256 checksums. A GitHub repository URL is not an archived DOI.
