"""Shared text-only Federalist parsing for both supplementary notebooks.

Source: https://www.gutenberg.org/cache/epub/18/pg18.txt
The first occurrence of essay 70 is kept. Numbers come from headings,
never list positions. Metadata and the shared PUBLIUS signature are removed.
"""

import re
from pathlib import Path

DISPUTED = set(range(49, 59)) | {62, 63}
JOINT = {18, 19, 20}
COLORS = {"Hamilton": "#aa3377", "Madison": "#dd9900", "Jay": "#4477aa", "Disputed": "#222222", "Joint": "#228833"}


def roman_number(value):
    """Convert the Roman or Arabic number in a Gutenberg essay heading."""
    if value.isdigit():
        return int(value)
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total, previous = 0, 0
    for character in reversed(value):
        number = values[character]
        total += -number if number < previous else number
        previous = max(previous, number)
    return total


def parse_federalist(text):
    """Return 85 numbered essay bodies and separate historical label metadata.

    Refuse unfamiliar layouts instead of silently shifting essay numbers.
    Disputed and jointly authored essays receive their own plot categories.
    These categories do not train a supervised classifier.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.split(r"\*\*\* END OF THE PROJECT GUTENBERG", text)[0]
    headings = list(re.finditer(r"(?m)^THE FEDERALIST\.\s+No\.\s+([IVXLCDM]+|[0-9]+)\.", text))
    essays = {}
    duplicates = []
    for i, heading in enumerate(headings):
        number = roman_number(heading.group(1))
        block = text[heading.end():headings[i + 1].start() if i + 1 < len(headings) else len(text)]
        greeting = re.search(r"To the People of the State of New York[:.]", block)
        if greeting is None:
            raise ValueError(f"No body delimiter in essay {number}.")
        header = block[:greeting.start()]
        names = re.findall(r"(?m)^(HAMILTON|MADISON|JAY)(?: (AND|OR) (HAMILTON|MADISON|JAY))?\s*$", header)
        if len(names) != 1:
            raise ValueError(f"Ambiguous author header in essay {number}.")
        body = re.split(r"(?m)^PUBLIUS\.\s*$", block[greeting.end():])[0].strip()
        if len(body) < 100:
            raise ValueError(f"Unexpectedly short essay {number}.")
        label = "Disputed" if number in DISPUTED else "Joint" if number in JOINT else names[0][0].capitalize()
        if number in essays:
            if number != 70:
                raise ValueError(f"Unexpected duplicate essay {number}.")
            duplicates.append(number)
            continue
        essays[number] = {"number": number, "author": label, "body": body, "source_author": " ".join(p for p in names[0] if p)}
    if set(essays) != set(range(1, 86)) or duplicates != [70]:
        raise ValueError("Expected essays 1-85 and exactly one duplicate version of essay 70. Check the source edition.")
    return [essays[number] for number in sorted(essays)]


def load_federalist(path="federalist.txt"):
    """Read the prepared Gutenberg file or explain the required setup step."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Run python prepare_data.py --federalist from the repository first.")
    return parse_federalist(path.read_text(encoding="utf-8-sig"))
