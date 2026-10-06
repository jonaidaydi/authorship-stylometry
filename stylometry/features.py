"""The Reuters word representation and 26 predefined style measurements.

The word tokenizer and settings preserve the Reuters exploration notebook.
The style measurements preserve the rounding used in the report analysis.
Neither representation is selected or tuned using test labels.
"""

import html
import re
import statistics
from collections import Counter

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

FUNCTION_WORDS = ["a", "and", "as", "at", "but", "for", "from", "in", "of", "on", "or", "that", "the", "to", "with"]
PUNCTUATION_NAMES = {",": "comma", ".": "period", ":": "colon", ";": "semicolon", "?": "question_mark", "!": "exclamation_mark"}
SURFACE = ["mean_sentence_length", "sentence_length_std", "mean_word_length", "type_token_ratio", "stopword_ratio"]
STYLE_COLUMNS = SURFACE + [f"function_{word}_per_1000" for word in FUNCTION_WORDS] + [f"punctuation_{name}_per_1000" for name in PUNCTUATION_NAMES.values()]
TOKEN_PATTERN = re.compile(
    r"[A-Za-z]{1,3}\$\d+(?:[.,]\d+)*|"
    r"\$\d+(?:[.,]\d+)*|"
    r"[€£¥]\d+(?:[.,]\d+)*|"
    r"[A-Za-z]+(?:-[A-Za-z]+)+|"
    r"[A-Za-z]{2,}"
)
CURRENCY_PATTERN = r"^[\$€£¥]"
stemmer = SnowballStemmer("english")


def custom_tokenizer_with_stemming(text):
    """Tokenise lowercase words and currency expressions, then stem words.

    Preserves the original notebook's treatment of currency prefixes and
    its English stop-word mismatch. The warning is documented, not hidden.
    """
    return [token if re.match(CURRENCY_PATTERN, token) else stemmer.stem(token)
            for token in TOKEN_PATTERN.findall(text.lower())]


def build_word_vectorizer():
    """Return an unfitted vectorizer with the shared, fixed Reuters settings."""
    return TfidfVectorizer(
        max_features=1000, use_idf=True, min_df=2, max_df=0.9,
        stop_words="english", tokenizer=custom_tokenizer_with_stemming,
        token_pattern=None, lowercase=False, ngram_range=(1, 3),
    )


def clean_text(text):
    """Normalise newlines and HTML entities, preserving case and punctuation."""
    return html.unescape(text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def safe_ratio(numerator, denominator):
    """Preserve the report's six-decimal ratio, returning zero for empty text."""
    return round(numerator / denominator, 6) if denominator else 0.0


def extract_document_features(doc):
    """Measure a spaCy document with the fixed 26-feature report definition.

    Only alphabetic tokens count as words. Sentence lengths omit sentences
    without alphabetic tokens. Punctuation counts exact spaCy punctuation
    tokens. Frequency values are the six-decimal ratio times 1,000, rounded
    to four decimals. POS n-grams are outside this experiment's scope.
    """
    tokens = [token for token in doc if not token.is_space]
    words = [token for token in tokens if token.is_alpha]
    lower = [token.lower_ for token in words]
    lengths = [sum(token.is_alpha for token in sentence) for sentence in doc.sents]
    lengths = [length for length in lengths if length]
    n = len(words)
    features = {
        "mean_sentence_length": round(statistics.mean(lengths), 4) if lengths else 0.0,
        "sentence_length_std": round(statistics.pstdev(lengths), 4) if lengths else 0.0,
        "mean_word_length": round(statistics.mean(len(token.text) for token in words), 4) if words else 0.0,
        "type_token_ratio": safe_ratio(len(set(lower)), n),
        "stopword_ratio": safe_ratio(sum(token.is_stop for token in words), n),
    }
    counts = Counter(lower)
    punctuation = Counter(token.text for token in tokens if token.is_punct)
    for word in FUNCTION_WORDS:
        features[f"function_{word}_per_1000"] = round(safe_ratio(counts[word], n) * 1000, 4)
    for mark, name in PUNCTUATION_NAMES.items():
        features[f"punctuation_{name}_per_1000"] = round(safe_ratio(punctuation[mark], n) * 1000, 4)
    return features
