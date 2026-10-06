# Reuters results and evidence

Generated with `python run_analysis.py` on 4 October 2026 in the tested Python 3.13.13 environment. The main README explains installation and the fixed 147-reference, 3-mystery and 150-test design.

## Files

| File | Contents |
|---|---|
| `analysis_summary.json` | Settings, versions, source hashes, metrics and explained variance |
| `document_manifest.csv` | The 300 corpus-relative filenames, roles and document hashes |
| `style_features.csv` | All 26 original style measurements for all documents |
| `standardisation.json` | Reference means, population deviations and retained columns |
| `author_centroids.csv` | The three author profiles in the standardised 25-dimensional space |
| `predictions.csv` | Every mystery/test prediction, distance and nearest reference document |
| `model_comparison.csv` | Accuracy and Macro-F1 for both evaluation groups |
| `*_confusion.csv` | True authors in rows, predicted authors in columns |
| `tfidf_author_profiles.csv` | Ten highest average TF-IDF weights per reference author |
| `*_pca_coordinates.csv` | Coordinates from reference-fitted full-SVD PCA |
| `matched_pca.png` | Reference articles and mystery stars in both representations |
| `qualitative_cases.json` | Reproducible case selection and feature-level diagnostics |

No full Reuters articles are redistributed here. The manifest lets readers locate each case in the original UCI corpus. All CSV encodings are UTF-8.

## Three qualitative cases

These cases were selected after the fixed evaluation. Within each outcome category, the alphabetically first corpus-relative test path was used. All three selected cases happen to be Patricia Commins articles. They illustrate particular decisions and are not a representative sample of all errors. No model or feature setting was changed after examining them.

### 1. Lexical error, correct style prediction

`C50test/PatriciaCommins/314014newsML.txt` discusses Dayton Hudson's store closures. The TF-IDF model predicts Tim Farrand, using `C50train/TimFarrand/173770newsML.txt`, a report on Burton's retail profits, as its closest reference.

The largest shared TF-IDF products are `store`, `retail`, `profit` and `divis`. Related retail vocabulary offers a plausible explanation for the lexical confusion across authors. Both style methods correctly predict Commins. Her centroid distance is 3.020, compared with 3.660 to Kazer and 3.729 to Farrand. This one successful style decision does not establish topic independence.

### 2. Correct lexical prediction, centroid error

`C50test/PatriciaCommins/334989newsML.txt` concerns a possible write-down of Quaker's Snapple business. Its nearest word-based reference, `C50train/PatriciaCommins/255733newsML.txt`, discusses a possible sale of the same business. The shared stems `quaker` and `snappl` dominate their lexical similarity.

TF-IDF and the style nearest neighbour correctly predict Commins. The style centroid predicts Kazer, with distances 4.079 to Kazer and 4.336 to Commins. In the exact squared-distance comparison, the stop-word ratio and mean word length favour Kazer, while some other features favour Commins. The centroid averages away variation between individual articles. The closely related subject matter also shows why lexical success should not be equated with pure personal style.

### 3. Agreement on the wrong author

`C50test/PatriciaCommins/404426newsML.txt` concerns CPC's proposed separation of its corn-refining business. All three methods predict Farrand. The TF-IDF neighbour, `C50train/TimFarrand/220852newsML.txt`, discusses Tate and Lyle, corn prices and its Staley subsidiary. Shared vocabulary includes `corn`, `sweeten` and `syrup`.

The style-centroid distances are 19.839 to Farrand and 20.030 to Commins. In the squared-distance margin between these authors, the semicolon frequency contributes +9.661 in favour of Farrand, the largest absolute contribution. The test article includes a product enumeration with this punctuation. This exposes sensitivity to a rare, standardised feature. It does not demonstrate a general stylistic rule for either author.

## Reading feature contributions

For the centroid winner `w` and an alternative `a`, the diagnostic calculates `(z_j - c_a,j)^2 - (z_j - c_w,j)^2` for each feature. A positive value favours the winner. For a wrong prediction, the alternative is the true author. For a correct prediction, it is the second-closest centroid. The JSON lists the eight largest absolute contributions. These are exact components of the model's distance comparison, not causal explanations of authorship or global feature importance.

Shared TF-IDF products are components of the inner product between the query and its nearest reference. Because the vectors are normalised, a larger total inner product corresponds to a smaller Euclidean distance.

## Limits

Test labels are used for evaluation and for the explicitly post-hoc choice of illustrative cases. They are not used in fitting, scaling, vocabulary selection or parameter selection. The results concern this fixed corpus and author trio. Exact-duplicate checks do not exclude related news or topic overlap.
