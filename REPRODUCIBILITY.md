# Reproduction and verification

## Verified environment

Verified on 4 October 2026 using Windows and Python 3.13.13. A new `.venv` was created in this repository and installed from `requirements.txt`. `pip check` reported no broken requirements. The fully resolved Windows environment is recorded in `requirements-lock.txt`. Other operating systems are not verified.

The pinned direct versions are NumPy 2.4.4, pandas 3.0.2, SciPy 1.18.1, scikit-learn 1.9.0, matplotlib 3.10.9, spaCy 3.8.14, `en_core_web_sm` 3.8.0, NLTK 3.10.3 and regex 2026.9.29. JupyterLab 4.4.3 and nbclient 0.10.2 support the notebooks.

## Checks performed

- Fresh package installation and dependency consistency check.
- Reuters download from the documented UCI URL and selection of the 300 required articles.
- Complete `run_analysis.py` execution using only this repository, its new environment and prepared corpus.
- Comparison with the preceding local calculation: `document_manifest.csv`, `style_features.csv`, `predictions.csv` and `model_comparison.csv` match, including all individual predictions.
- Four semantic tests covering word/sentence counts, empty documents, reference-only scaling, nearest-neighbour ties and minimal cleaning.
- The three original notebooks are preserved byte for byte from commit `64b1f8a`. Their execution is outside this verification of the complementary Reuters analysis.

These checks verify the implementation for the supplied data and fixed settings. They do not establish generalisation to other authors or topic-independent attribution.

## Data preparation

The downloader extracts only the selected Reuters author folders and checks archive paths before writing. It supports the UCI ZIP and an existing local archive. The reference archive used for the original report calculation has SHA-256 `d1de4af0f439c9ffd097dc30ea15fb22cb015e1cbfde986bb5da0622904dbd0d`. Individual byte and stripped-text hashes in `results/document_manifest.csv` identify the actual analysis inputs even if archive packaging changes.

The local `.cache/data_sources.json` records download URLs and archive hashes. No machine-specific absolute paths are written into the published result tables. Raw corpora and package caches are excluded from Git.

## Precise feature conventions

- Word counts use spaCy `is_alpha` tokens. Non-alphabetic currency and number tokens do not contribute to style word counts.
- Sentence length counts alphabetic tokens and omits sentences with no such tokens. The standard deviation is the population standard deviation.
- Mean sentence length, its standard deviation and mean word length are rounded to four decimals.
- Type-token ratio and stop-word ratio are rounded to six decimals. Function-word and punctuation frequencies are the ratio rounded to six decimals, multiplied by 1,000 and rounded to four decimals. This preserves the earlier calculation.
- Punctuation counts exact punctuation tokens, not every character within arbitrary tokens.
- The constant-feature filter and scaling parameters use reference documents only. Exclamation-mark frequency is constant there and is removed.
- Style nearest neighbours and centroids use all 25 retained dimensions. TF-IDF nearest neighbours use all 1,000 features.
- PCA uses `svd_solver="full"`. Its signs may change on another numerical platform without changing the represented geometry.
- UTF-8 decoding with `errors="ignore"` follows the existing Reuters notebook. This can discard invalid bytes and is a limitation of the preserved preprocessing.

## Deliberately preserved warning

scikit-learn warns that the English stop-word list is not fully aligned with the Snowball tokenizer. This warning is expected and does not stop execution. Changing the stop-word handling would change the word representation. It was therefore preserved for the matched comparison and documented in the reports.

## Scope and original notebooks

This verification covers Jonaid Aydi's independent Reuters comparison in `run_analysis.py` and its supporting modules. It does not certify the execution or methodology of Kai S. Kurono's original notebooks. Those files have been restored to commit `64b1f8a`, including their outputs. The complementary analysis does not import or execute them.

The original Federalist notebooks retain their own preprocessing, essay-number handling and author categories. Their metadata filtering and the square-distance input in the HCA remain matters for Kai to review. No corrected Federalist parser or replacement figures form part of the complementary contribution.

## Course connection

The implementation uses dictionaries and comprehensions for corpus records, functions with docstrings, a virtual environment and pinned dependencies, argparse options, CSV/JSON outputs, NLTK stemming, spaCy token/sentence processing, pandas tables and NumPy distance calculations. Git and bilingual READMEs support a shared, inspectable project. These are concrete connections to the supplied course materials. The project does not attempt to demonstrate every topic, and no custom class is needed for this small workflow.

The original project suggestion includes POS n-grams among possible style features. This fixed comparison uses the documented 26 measurements. Adding another feature family after inspecting test results would be a new experiment requiring a clearly separated evaluation.
