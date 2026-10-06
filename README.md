# Automatic Authorship Attribution of Journalistic Texts Using Stylometric Features and Classical Machine Learning Methods

[Deutsch](README_de.md) | **English**

Course project by **Jonaid Aydi and Kai S. Kurono**, *Advanced Python for NLP* (2026), Rainer Osswald and Yulia Zinova.

Can word-based and explicit style representations distinguish Reuters articles by three authors? We compare a TF-IDF nearest neighbour with a style nearest neighbour and a style author centroid. PCA and hierarchical clustering support exploration. They do not provide the classifier input.

## Results and reports

| Method | Mystery correct | Official test correct | Test accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| TF-IDF nearest neighbour | 3/3 | 137/150 | 91.33% | 0.9138 |
| Style nearest neighbour | 1/3 | 104/150 | 69.33% | 0.6864 |
| Style author centroid | 2/3 | 124/150 | 82.67% | 0.8283 |

The test contains 50 articles per author. A constant single-author prediction scores 33.33% accuracy. These results concern one author trio and do not establish topic-independent authorship recognition.

- [Joint report](reports/joint_report.pdf), intended as the combined submission report.
- Individual contributions: [Kai S. Kurono](reports/kai_report.pdf) and [Jonaid Aydi](reports/jonaid_report.pdf).
- [German presentation](presentation/ap_presentation_de.pptx) and [speaker notes](presentation/speaker_notes_de.md).
- [Result tables and document-level evidence](results/README.md).
- [Verification record and technical details](REPRODUCIBILITY.md).

![Matched PCA views of the same reference and mystery texts](results/matched_pca.png)

## Install and run

Tested on **Windows with Python 3.13.13**, using a newly created virtual environment. Run these commands from the repository root. Installation and initial data preparation require internet access.

```powershell
git clone https://github.com/jonaidaydi/authorship-stylometry.git
cd authorship-stylometry
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe prepare_data.py
.venv\Scripts\python.exe run_analysis.py
```

Use a Python 3.13 interpreter for `python -m venv`. On macOS/Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python`. Those operating systems have not been tested.

`requirements.txt` pins direct dependencies and includes `en_core_web_sm` 3.8.0. `requirements-lock.txt` records the complete tested Windows environment. The model wheel downloads from the official spaCy GitHub release. No NLTK corpus download is needed for Snowball stemming.

`prepare_data.py` downloads the UCI ZIP and extracts only the 300 selected articles to `reuter+50+50/`. `--federalist` also prepares `federalist.txt` for the supplementary notebooks. Existing files with different contents are not overwritten. To use an existing Reuters ZIP:

```powershell
.venv\Scripts\python.exe prepare_data.py --archive "path/to/reuter_50_50.zip"
```

`run_analysis.py` is the main entry point. It writes the comparison, predictions, feature tables, hashes, PCA coordinates and figure to `results/`. To keep the published reference outputs intact during your own run:

```powershell
.venv\Scripts\python.exe run_analysis.py --output-dir verification_outputs/reuters
```

Use `--data-dir` when the corpus is stored elsewhere. The folder must contain `C50train` and `C50test`, each with `WilliamKazer`, `TimFarrand` and `PatriciaCommins` subfolders containing 50 `.txt` files. Raw data, caches and virtual environments are ignored by Git.

## Experimental design

| Author | Corpus folder | Reference | Mystery | Official test |
|---|---|---:|---:|---:|
| William Kazer | `WilliamKazer` | 49 | 1 | 50 |
| Tim Farrand | `TimFarrand` | 49 | 1 | 50 |
| Patricia Commins | `PatriciaCommins` | 49 | 1 | 50 |
| Total | | 147 | 3 | 150 |

The first sorted training filename per author is the mystery document. The remaining 49 training articles form the reference set. The mystery texts are not added back for the official test evaluation. All methods use the same documents.

TF-IDF uses NLTK Snowball stemming, word n-grams of length 1–3, at most 1,000 features, `min_df=2`, `max_df=0.9`, and English stop words. The fitted vocabulary has 832 unigrams, 151 bigrams and 17 trigrams. The original stemmer/stop-word mismatch produces a documented scikit-learn warning. We preserve it to keep the comparison consistent with the exploration.

The 26 predefined style features comprise five surface measurements, frequencies of 15 function words and frequencies of six punctuation marks. Alphabetic spaCy tokens form the word counts. One constant feature is removed, leaving 25 dimensions. The shared implementation preserves the report's rounding rules. POS n-grams are not part of this experiment.

Vocabulary, IDF, scaling, constant-feature removal, author centroids and PCA are fitted only on the reference articles. NumPy computes Euclidean distances. The full-SVD PCA is deterministic and used only for display. No feature group or model parameter was selected from the official test scores.

## Code and notebooks

| File | Responsibility |
|---|---|
| `prepare_data.py` | Download and select the data |
| `run_analysis.py` | Complete Reuters comparison, evaluation and outputs |
| `stylometry/corpus.py` | Document identities, split and exact-duplicate checks |
| `stylometry/features.py` | Shared word tokenizer and 26 style measurements |
| `stylometry/models.py` | Reference scaling and NumPy nearest-neighbour distances |
| [Reuters notebook](reuter%20HKA%2C%20HCA.ipynb) | TF-IDF exploration, full-space Ward HCA, PCA and three mystery predictions |
| [Federalist PCA](HKA%20Federalist%20Papiere.ipynb) | Supplementary TF-IDF/PCA example |
| [Federalist HCA](HCA%20Federalist%20Papiere.ipynb) | Supplementary word-frequency HCA with IDF disabled |

The three notebooks are Kai's original contribution from commit `64b1f8a`, preserved with their stored outputs. Open them with `.venv\Scripts\python.exe -m jupyterlab`. The Reuters notebook contains its own seaborn installation cell. Changes to these notebooks and their methodology remain with Kai.

Jonaid's complementary Reuters analysis runs independently through `run_analysis.py`. It reproduces the word representation and document split as a comparison baseline in separate Python modules. The installation instructions, result files and checks below refer to this complementary analysis.

The Federalist notebooks are supplementary historical examples. Their original preprocessing and author categories have not been harmonised. Headings and author metadata can enter their features. The HCA passes a square distance matrix to `linkage`, which treats its rows as observations. These outputs are separate from the complementary Reuters test evaluation.

## Checks

```powershell
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The four tests check Jonaid's complementary analysis. The main analysis has been compared with the earlier local report calculation: document identities, all feature values, predictions and model metrics agree.

## Interpretation and limitations

Topics, editorial conventions and related news stories may contribute to the scores. Exact byte/text duplicate checks are not a near-duplicate or topic-control analysis. Three mystery examples alone cannot measure general performance. Standardisation can give rare punctuation a large influence, and type-token ratio depends on text length. The [three documented error cases](results/README.md) illustrate these limitations. They were selected after evaluation and did not change the models.

## Contributions and tools

| Contributor | Project responsibility |
|---|---|
| Kai S. Kurono | Reuters and Federalist exploration notebooks, word representations, PCA, hierarchical clustering and Reuters mystery attribution. |
| Jonaid Aydi | Complementary style-feature comparison, official test evaluation, repository integration, reproducibility and bilingual documentation. |

Codex assisted with Jonaid's complementary analysis, its integration and verification, and preparation of the supporting materials. Kai's three original notebooks remain unchanged. The joint report and slides are proposals for both contributors to review together. The repository records the implemented methods and reproducible outputs. The two contributors remain responsible for the submitted work and its explanation.

## Sources

- Liu, Z. (2006). *Reuter_50_50*. UCI Machine Learning Repository. [DOI: 10.24432/C5DS42](https://doi.org/10.24432/C5DS42), dataset page specifies CC BY 4.0.
- Hamilton, Madison and Jay. *The Federalist Papers*. [Project Gutenberg eBook 18](https://www.gutenberg.org/ebooks/18).
- [NLTK Snowball](https://www.nltk.org/api/nltk.stem.snowball.html), [spaCy linguistic features](https://spacy.io/usage/linguistic-features), [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [SciPy linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).
