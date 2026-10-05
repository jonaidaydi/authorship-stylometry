# Automatic Authorship Attribution of Journalistic Texts Using Stylometric Features and Classical Machine Learning Methods

[Deutsch](README_de.md) | **English**

Course project by **Jonaid Aydi and Kai S. Kurono**, *Advanced Python for NLP* (2026),
Rainer Osswald and Yulia Zinova.

The project explores how texts by different authors are distributed in
word-based representations. Principal component analysis (PCA, German HKA)
and hierarchical cluster analysis (HCA) make similarities and overlaps visible.
The Federalist Papers and Reuters articles provide two case studies.

## Notebooks

| Notebook | Analysis |
|---|---|
| [HKA Federalist Papiere.ipynb](HKA%20Federalist%20Papiere.ipynb) | TF-IDF, two-dimensional PCA, essay numbers and word loadings |
| [HCA Federalist Papiere.ipynb](HCA%20Federalist%20Papiere.ipynb) | Hierarchical clustering of word-frequency vectors with IDF disabled |
| [reuter HKA, HCA.ipynb](reuter%20HKA%2C%20HCA.ipynb) | Stemming, TF-IDF word n-grams, PCA, hierarchical clustering and nearest-text attribution |

The Reuters notebook currently selects William Kazer, Tim Farrand and
Patricia Commins. For each author, it reserves the first file in sorted order
as a mystery text and uses the remaining 49 articles for the reference set.
The vectoriser and PCA are fitted on those 147 reference articles.
The three reserved texts are then transformed using the fitted models.
Attribution uses the closest reference text in the full TF-IDF space.

## Install and open

Use Python 3 and run the following commands from this repository folder.
Installation and the initial corpus download require internet access.

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m jupyterlab
```

On macOS or Linux, use `.venv/bin/python` for the last two commands.

Open a notebook in JupyterLab and run all cells from top to bottom with a
fresh kernel. Keep the notebooks in the repository root because their data
paths are relative to that folder. The Reuters notebook includes an extra
seaborn installation cell, although seaborn is already in `requirements.txt`.

The dependency list follows the current imports. A fresh installation and
complete execution of all three notebooks have not yet been verified for
this repository snapshot. Package versions are not pinned yet.

## Prepare the data

### Federalist Papers

Each Federalist notebook downloads the
[Project Gutenberg text](https://www.gutenberg.org/cache/epub/18/pg18.txt)
to `federalist.txt` in its first code cell. The PCA notebook also writes
individual essays into `federalist_papers/`. Both locations are ignored by Git.

### Reuters

Download [Reuter_50_50 from UCI](https://archive.ics.uci.edu/dataset/217/reuter%2B50%2B50)
using the dataset page's Download button, or use the
[ZIP archive](https://archive.ics.uci.edu/static/public/217/reuter%2B50%2B50.zip).
Extract its contents into a folder named exactly `reuter+50+50` beside the notebooks.

The resulting paths must include:

```text
reuter+50+50/
  C50train/
    WilliamKazer/
    TimFarrand/
    PatriciaCommins/
    ... other authors
  C50test/
    ... authors
```

The current Reuters notebook reads `C50train` only. Its three mystery texts
are reserved from that split. It does not evaluate the official `C50test` split.
The downloaded corpus and ZIP archive stay local.

## Read the figures and results

The Federalist notebooks display PCA plots and a dendrogram.
The Reuters notebook also writes `hca_dendrogram.png` and
`hka_pca_biplot.png`, then prints its three attribution results and the
highest-weight features for each author. Generated images are ignored by Git.

PCA displays only two dimensions. Its axes report the fraction of variance
shown. Visible separation can reflect topic-related words as well as author
differences. Three mystery texts are illustrative examples and do not establish
general classification accuracy.

## Current methodological checks

The notebooks preserve the current exploration. Before treating the results
as a final evaluation, the following points need to be resolved:

- The Federalist feature vectors still include headings and author metadata.
  These need to be separated from the essay bodies.
- The two Federalist notebooks use different rules for disputed authorship.
  Their labels and treatment of the duplicate essay 70 need to be aligned
  and documented.
- The Federalist HCA notebook passes a square distance matrix to Ward linkage.
  SciPy interprets its rows as observations. For clustering the original word
  vectors, it needs the observation matrix or condensed Euclidean distances.
  See the [SciPy linkage documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).

## Contributions

| Contributor | Contribution |
|---|---|
| Kai S. Kurono | Federalist and Reuters exploration notebooks, word representations, PCA, hierarchical clustering and Reuters mystery-text attribution. |
| Jonaid Aydi | Repository organisation, dependency list, installation instructions and English and German documentation. |

## Data sources

- Hamilton, Madison and Jay. *The Federalist Papers*.
  [Project Gutenberg, eBook 18](https://www.gutenberg.org/ebooks/18).
- Liu, Z. (2006). *Reuter_50_50*. UCI Machine Learning Repository.
  [DOI: 10.24432/C5DS42](https://doi.org/10.24432/C5DS42).
  The UCI dataset page specifies CC BY 4.0.
