"""Download Reuters and optionally Federalist, keeping raw texts out of Git."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
REUTERS_URL = "https://archive.ics.uci.edu/static/public/217/reuter%2B50%2B50.zip"
FEDERALIST_URL = "https://www.gutenberg.org/cache/epub/18/pg18.txt"
AUTHORS = {"WilliamKazer", "TimFarrand", "PatriciaCommins"}


def extract_selected(archive_bytes, destination):
    """Extract only the 300 selected articles, including nested ZIP archives.

    Member paths are validated before writing. No archive-supplied absolute
    path or parent-directory traversal can become an output path.
    """
    count = 0
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            if member.is_dir():
                continue
            if path.suffix.lower() == ".zip":
                count += extract_selected(archive.read(member), destination)
                continue
            parts = path.parts
            if len(parts) == 3 and parts[0] in {"C50train", "C50test"} and parts[1] in AUTHORS and path.suffix == ".txt":
                target = destination.joinpath(*parts)
                data = archive.read(member)
                if target.exists() and target.read_bytes() != data:
                    raise ValueError(f"Existing corpus file differs: {target}. Choose another --data-dir.")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an existing UCI ZIP instead of downloading")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "reuter+50+50")
    parser.add_argument("--federalist", action="store_true", help="Also download the historical supplementary corpus")
    args = parser.parse_args()
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    archive_path = args.archive or cache / "reuter_50_50.zip"
    if not archive_path.exists():
        if args.archive:
            raise FileNotFoundError(archive_path)
        print("Downloading Reuters from UCI...", flush=True)
        with urllib.request.urlopen(REUTERS_URL, timeout=120) as response:
            archive_path.write_bytes(response.read())
    raw = archive_path.read_bytes()
    count = extract_selected(raw, args.data_dir)
    if count != 300:
        raise ValueError(f"Expected 300 selected articles, extracted {count}.")
    record = {"reuters_url": REUTERS_URL, "archive_sha256": hashlib.sha256(raw).hexdigest(), "selected_articles": count}
    if args.federalist:
        target = ROOT / "federalist.txt"
        if not target.exists():
            with urllib.request.urlopen(FEDERALIST_URL, timeout=120) as response:
                target.write_bytes(response.read())
        record.update(federalist_url=FEDERALIST_URL, federalist_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (cache / "data_sources.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Prepared {count} Reuters articles. Raw data stay local.")


if __name__ == "__main__":
    main()
