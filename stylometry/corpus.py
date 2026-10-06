"""Load the fixed Reuters experiment and record its exact document identities."""

import hashlib
from pathlib import Path

import pandas as pd

AUTHORS = ["WilliamKazer", "TimFarrand", "PatriciaCommins"]
NAMES = ["William Kazer", "Tim Farrand", "Patricia Commins"]


def load_reuters(corpus):
    """Read 300 articles with the notebook's 147/3/150 split.

    The first alphabetically sorted training file per author is held out.
    Raw byte and stripped-text hashes detect exact reference/evaluation
    duplicates. Related stories and topic overlap are not detected.
    """
    rows, texts = [], []
    for split in ["C50train", "C50test"]:
        for author in AUTHORS:
            files = sorted((Path(corpus) / split / author).glob("*.txt"))
            if len(files) != 50:
                raise ValueError(f"Expected 50 articles in {split}/{author}, found {len(files)}. Run prepare_data.py first.")
            for i, path in enumerate(files):
                raw = path.read_bytes()
                # Preserves the existing notebook's decoding rule.
                text = raw.decode("utf-8", errors="ignore")
                role = "official_test" if split == "C50test" else "mystery" if i == 0 else "reference"
                rows.append({"author": author, "document": path.name,
                             "relative_path": path.relative_to(corpus).as_posix(), "role": role,
                             "sha256": hashlib.sha256(raw).hexdigest(),
                             "text_sha256": hashlib.sha256(text.strip().encode()).hexdigest()})
                texts.append(text)
    frame = pd.DataFrame(rows)
    for role in ["mystery", "official_test"]:
        for column in ["sha256", "text_sha256"]:
            overlap = set(frame.loc[frame.role.eq("reference"), column]) & set(frame.loc[frame.role.eq(role), column])
            if overlap:
                raise ValueError(f"Exact reference/{role} duplicate detected with {column}.")
    return frame, texts
