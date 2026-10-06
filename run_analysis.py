"""Reproduce the joint Reuters comparison from the repository alone.

Run ``python prepare_data.py`` once, then ``python run_analysis.py``.
All learned transformations use the 147 reference articles exclusively.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from stylometry.corpus import AUTHORS, NAMES, load_reuters
from stylometry.features import STYLE_COLUMNS, build_word_vectorizer, clean_text, extract_document_features
from stylometry.models import fit_scaler, nearest, transform_style

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    """Write readable, portable JSON without machine-specific paths."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def evaluate(frame, representations, centroids, output):
    """Evaluate both held-out groups and export every prediction and matrix."""
    reference = frame.role.eq("reference").to_numpy()
    labels = frame.author.to_numpy()
    metrics, predictions = [], []
    for role in ["mystery", "official_test"]:
        mask = frame.role.eq(role).to_numpy()
        for method in ["tfidf_1nn", "style_1nn", "style_centroid"]:
            if method == "style_centroid":
                predicted, distances, indices = nearest(centroids, representations["style"][mask], AUTHORS)
            else:
                values = representations["tfidf" if method == "tfidf_1nn" else "style"]
                predicted, distances, indices = nearest(values[reference], values[mask], labels[reference])
            metrics.append({"role": role, "model": method, "n": int(mask.sum()),
                            "correct": int((predicted == labels[mask]).sum()),
                            "accuracy": float(accuracy_score(labels[mask], predicted)),
                            "macro_f1": float(f1_score(labels[mask], predicted, labels=AUTHORS, average="macro", zero_division=0))})
            for source, prediction, distance, neighbour in zip(np.flatnonzero(mask), predicted, distances, indices):
                row = frame.iloc[source].to_dict()
                row.update(model=method, predicted_author=prediction, distance=float(distance))
                row["reference_document"] = "author centroid" if method == "style_centroid" else frame.loc[reference].iloc[neighbour].relative_path
                predictions.append(row)
            pd.DataFrame(confusion_matrix(labels[mask], predicted, labels=AUTHORS), index=AUTHORS, columns=AUTHORS).to_csv(output / f"{role}_{method}_confusion.csv")
    pd.DataFrame(metrics).to_csv(output / "model_comparison.csv", index=False)
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(output / "predictions.csv", index=False)
    return metrics, predictions


def case_diagnostics(frame, words, scaled, centroids, vocabulary, retained, predictions, output):
    """Select three post-hoc error cases by filename, without model tuning.

    Selection categories are lexical failure/style success, the reverse,
    and failure of both. Cases illustrate mechanisms, not prevalence.
    Squared-distance contributions describe the centroid decision exactly.
    """
    test = predictions.loc[predictions.role.eq("official_test")]
    wide = test.pivot(index="relative_path", columns="model", values="predicted_author")
    true = frame.set_index("relative_path").author.reindex(wide.index)
    lexical_ok = wide.tfidf_1nn.eq(true)
    style_ok = wide.style_centroid.eq(true)
    categories = {"lexical_error_style_correct": ~lexical_ok & style_ok,
                  "lexical_correct_style_error": lexical_ok & ~style_ok,
                  "both_incorrect": ~lexical_ok & ~style_ok}
    positions = {p: i for i, p in enumerate(frame.relative_path)}
    cases = []
    for category, mask in categories.items():
        candidates = wide.loc[mask].sort_index()
        if candidates.empty:
            continue
        path = candidates.index[0]
        i = positions[path]
        truth = frame.iloc[i].author
        nearest_row = test.loc[test.relative_path.eq(path) & test.model.eq("tfidf_1nn")].iloc[0]
        neighbour = positions[nearest_row.reference_document]
        shared = words[i] * words[neighbour]
        top = np.argsort(shared)[-8:][::-1]
        distances = np.linalg.norm(centroids - scaled[i], axis=1)
        best = int(np.argmin(distances))
        true_index = AUTHORS.index(truth)
        other = true_index if best != true_index else int(np.argsort(distances)[1])
        contribution = (scaled[i] - centroids[other]) ** 2 - (scaled[i] - centroids[best]) ** 2
        important = np.argsort(np.abs(contribution))[-8:][::-1]
        cases.append({"category": category, "document": path, "true_author": truth,
                      "predictions": candidates.iloc[0].to_dict(),
                      "tfidf_neighbour": nearest_row.reference_document,
                      "shared_tfidf_features": [{"feature": str(vocabulary[j]), "product": float(shared[j])} for j in top if shared[j] > 0],
                      "centroid_distances": dict(zip(AUTHORS, distances.tolist())),
                      "centroid_comparison": {"winner": AUTHORS[best], "alternative": AUTHORS[other]},
                      "style_distance_contributions": [{"feature": str(retained[j]), "alternative_squared_distance_minus_winner": float(contribution[j])} for j in important]})
    write_json(output / "qualitative_cases.json", cases)


def plot_pca(frame, representations, output):
    """Fit full-SVD PCA on references and display the matched mystery demo."""
    reference = frame.role.eq("reference").to_numpy()
    mystery = frame.role.eq("mystery").to_numpy()
    labels = frame.author.to_numpy()
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.7), constrained_layout=True)
    for ax, (name, matrix) in zip(axes, representations.items()):
        pca = PCA(n_components=2, svd_solver="full").fit(matrix[reference])
        projection = pca.transform(matrix)
        variance = pca.explained_variance_ratio_
        results[name] = {"explained_variance_ratio": variance.tolist(), "features": matrix.shape[1], "solver": "full", "fit_documents": 147}
        coords = frame.copy()
        coords[["pc1", "pc2"]] = projection
        coords.to_csv(output / f"{name}_pca_coordinates.csv", index=False)
        for author, display, color in zip(AUTHORS, NAMES, ["#c12c89", "#dd9300", "#3889ac"]):
            selection = reference & (labels == author)
            ax.scatter(projection[selection, 0], projection[selection, 1], s=20, color=color, alpha=.65, label=display)
            selection = mystery & (labels == author)
            ax.scatter(projection[selection, 0], projection[selection, 1], s=155, color=color, marker="*", edgecolor="#222222", linewidth=.7)
        title = "Word TF-IDF (1,000 features)" if name == "tfidf" else f"Style ({matrix.shape[1]} variable features)"
        ax.set(title=title, xlabel=f"PC1 ({variance[0]:.2%})", ylabel=f"PC2 ({variance[1]:.2%})")
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Same 147 reference articles and three mystery texts (stars)", fontsize=12)
    fig.savefig(output / "matched_pca.png", dpi=220)
    plt.close(fig)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "reuter+50+50")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame, texts = load_reuters(args.data_dir)
    reference = frame.role.eq("reference").to_numpy()
    labels = frame.author.to_numpy()
    frame.to_csv(output / "document_manifest.csv", index=False)

    vectorizer = build_word_vectorizer()
    vectorizer.fit([text for text, keep in zip(texts, reference) if keep])
    words = vectorizer.transform(texts).toarray()
    vocabulary = vectorizer.get_feature_names_out()
    print(f"Word vectors: {words.shape}", flush=True)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as error:
        raise RuntimeError("Install all packages and the model with pip install -r requirements.txt.") from error
    features = []
    for i, doc in enumerate(nlp.pipe((clean_text(t) for t in texts), batch_size=16), 1):
        features.append(extract_document_features(doc))
        if i % 50 == 0:
            print(f"Style features: {i}/300", flush=True)
    feature_frame = pd.DataFrame(features)[STYLE_COLUMNS]
    style = feature_frame.to_numpy(float)
    mean, std, variable = fit_scaler(style[reference])
    scaled = transform_style(style, mean, std, variable)
    retained = np.array(STYLE_COLUMNS)[variable]
    pd.concat([frame, feature_frame], axis=1).to_csv(output / "style_features.csv", index=False)
    write_json(output / "standardisation.json", {"features": STYLE_COLUMNS, "training_mean": mean.tolist(), "training_std": std.tolist(), "retained": retained.tolist()})
    centroids = np.array([scaled[reference & (labels == author)].mean(axis=0) for author in AUTHORS])
    pd.DataFrame(centroids, index=AUTHORS, columns=retained).to_csv(output / "author_centroids.csv")
    profiles = []
    for author in AUTHORS:
        weights = words[reference & (labels == author)].mean(axis=0)
        for j in np.argsort(weights)[-10:][::-1]:
            profiles.append({"author": author, "feature": str(vocabulary[j]), "mean_tfidf": float(weights[j])})
    pd.DataFrame(profiles).to_csv(output / "tfidf_author_profiles.csv", index=False)
    representations = {"tfidf": words, "style": scaled}
    metrics, predictions = evaluate(frame, representations, centroids, output)
    case_diagnostics(frame, words, scaled, centroids, vocabulary, retained, predictions, output)
    pca = plot_pca(frame, representations, output)
    sources = [Path("run_analysis.py")] + [p.relative_to(ROOT) for p in sorted((ROOT / "stylometry").glob("*.py"))]
    summary = {
        "authors": AUTHORS, "reference_articles": 147, "mystery_articles": 3, "official_test_articles": 150,
        "duplicate_check": "No exact raw-byte or stripped-text matches from reference to either evaluation group",
        "word_features": len(vocabulary), "unigrams": int(sum(" " not in v for v in vocabulary)),
        "bigrams": int(sum(v.count(" ") == 1 for v in vocabulary)), "trigrams": int(sum(v.count(" ") == 2 for v in vocabulary)),
        "style_features": STYLE_COLUMNS, "style_variable_features": int(variable.sum()),
        "style_constant_features": np.array(STYLE_COLUMNS)[~variable].tolist(), "pca": pca, "metrics": metrics,
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "spacy", "en_core_web_sm", "nltk", "regex", "matplotlib"]},
        "source_sha256": {p.as_posix(): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "case_selection": "Post-hoc examples selected by category and sorted relative path. No refitting or tuning.",
    }
    write_json(output / "analysis_summary.json", summary)
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
