"""
Keyword extraction tool for AI agents.

This module provides a small, dependency-light **keyword extraction** utility that can be
used by AI agents to quickly identify the most salient terms in a piece of text.

## Purpose
Keyword extraction is useful for:
- **Tagging / labeling**: generate topic-like labels from raw text
- **Indexing / retrieval**: produce compact terms for search queries and filtering
- **Agent workflows**: reduce long inputs into a small set of representative concepts

## Approach (TF‑IDF style)
The extractor scores tokens using a TF‑IDF style heuristic:
- Split the input into sentences and treat each sentence as a small “document”.
- Compute **term frequency (TF)** across the full input.
- Compute **document frequency (DF)** across sentences.
- Score terms with \(TF \\times IDF\), where \(IDF = \\log\\frac{N+1}{DF+1} + 1\).

NLTK is used when available for sentence/word tokenization and English stopword removal.
If NLTK is not installed, a regex-based fallback tokenizer is used and stopwords are not
filtered.

## Outputs
The main function, `extract_keywords`, returns a **JSON-serializable** dictionary:
`{\"keywords\": [{\"word\": str, \"score\": float}, ...], \"metadata\": {...}}`.

For agent integration, a lightweight `Tool` wrapper is provided and instantiated as
`keyword_extractor`.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

# Optional: use nltk if available; fallback to simple tokenization
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


def _ensure_nltk_data() -> None:
    """
    Attempt to make required NLTK resources available.

    Notes:
        - This function is best-effort: it will not raise if downloads fail.
        - Network access may be unavailable in some environments; in that case the
          extractor still works via the regex fallback tokenizer.
    """
    if not NLTK_AVAILABLE:
        return
    for resource in ("punkt", "stopwords"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


def _simple_tokenize(text: str) -> list[str]:
    """
    Tokenize text into lowercase “word” tokens using regex.

    This is used when NLTK is not available or when NLTK tokenization fails.

    Args:
        text: Input string.

    Returns:
        A list of lowercase alphanumeric tokens.
    """
    text_lower = text.lower()
    words = re.findall(r"\b[a-z0-9]+\b", text_lower)
    return words


def _simple_sentences(text: str) -> list[str]:
    """
    Split text into sentence-like chunks using regex.

    This is a lightweight approximation used when NLTK sentence tokenization is not
    available.

    Args:
        text: Input string.

    Returns:
        A list of non-empty sentence strings.
    """
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def extract_keywords(
    text: str,
    top_n: int = 10,
    min_word_length: int = 2,
    use_stopwords: bool = True,
) -> dict[str, Any]:
    """
    Extract the most important keywords from the given text.

    The extractor uses NLTK for tokenization and stopword removal (when available),
    then scores terms using a TF‑IDF style approach. Concretely, the input text is split
    into sentences and treated as a small set of documents so that terms that appear
    in many sentences are down-weighted relative to more discriminative terms.

    This function is designed to be safe to call from agent code:
    - validates inputs and raises clear exceptions for invalid types/values
    - returns only JSON-serializable types (dict/list/str/int/float/bool)

    Args:
        text: The input text to extract keywords from. Must be a non-empty string.
        top_n: The maximum number of keywords to return. Must be a positive integer.
        min_word_length: Minimum character length for a token to be considered.
            Must be non-negative. (Applies after lowercasing / tokenization.)
        use_stopwords: If True (and NLTK is available), filter out common
            stopwords (e.g. "the", "is") from candidates.

    Returns:
        A JSON-serializable dictionary with:

        - **keywords**: list of keyword objects, sorted by descending importance:
          `[{\"word\": <token>, \"score\": <tfidf_score>}, ...]`
        - **metadata**: information about the run:
          `{\"total_terms\": <unique_terms_scored>, \"top_n\": <requested_top_n>, \"nltk_used\": <bool>}`

    Raises:
        TypeError:
            - if `text` is not a string
            - if `top_n` or `min_word_length` are not integers
            - if `use_stopwords` is not a boolean
        ValueError:
            - if `text` is empty or whitespace-only
            - if `top_n < 1`
            - if `min_word_length < 0`

    Examples:

        Basic usage:

        >>> extract_keywords(\"Python is great for data science.\", top_n=3)\n
        Using the `Tool` wrapper:

        >>> keyword_extractor.execute(text=\"Some text...\", top_n=5)
    """
    # --- Input validation ---
    if not isinstance(text, str):
        raise TypeError("'text' must be a string")

    if not text or not text.strip():
        raise ValueError("'text' cannot be empty or whitespace-only")

    if not isinstance(top_n, int):
        raise TypeError("'top_n' must be an integer")

    if top_n < 1:
        raise ValueError("'top_n' must be at least 1")

    if not isinstance(min_word_length, int):
        raise TypeError("'min_word_length' must be an integer")

    if min_word_length < 0:
        raise ValueError("'min_word_length' cannot be negative")

    if not isinstance(use_stopwords, bool):
        raise TypeError("'use_stopwords' must be a boolean")

    # --- Setup ---
    _ensure_nltk_data()

    if NLTK_AVAILABLE:
        try:
            stop = set(stopwords.words("english"))
        except Exception:
            stop = set()
    else:
        stop = set()

    # --- Tokenize into sentences then words ---
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = _simple_sentences(text)
    else:
        sentences = _simple_sentences(text)

    if not sentences:
        sentences = [text]

    doc_tokens: list[list[str]] = []
    for sent in sentences:
        if NLTK_AVAILABLE:
            try:
                words = word_tokenize(sent.lower())
            except Exception:
                words = _simple_tokenize(sent)
        else:
            words = _simple_tokenize(sent)

        words = [
            w
            for w in words
            if len(w) >= min_word_length
            and w.isalnum()
            and (not use_stopwords or w not in stop)
        ]
        doc_tokens.append(words)

    # --- TF-IDF over sentences as documents ---
    num_docs = len(doc_tokens)
    doc_freq: dict[str, int] = {}
    term_freq: dict[str, float] = {}

    for tokens in doc_tokens:
        seen_in_doc: set[str] = set()
        for t in tokens:
            term_freq[t] = term_freq.get(t, 0.0) + 1.0
            seen_in_doc.add(t)
        for t in seen_in_doc:
            doc_freq[t] = doc_freq.get(t, 0) + 1

    # TF-IDF score: tf * idf, with idf = log((N+1)/(df+1)) + 1
    scores: dict[str, float] = {}
    for term, tf in term_freq.items():
        df = doc_freq.get(term, 0)
        idf = math.log((num_docs + 1) / (df + 1)) + 1.0
        scores[term] = tf * idf

    # Sort by score descending, then alphabetically for ties
    sorted_terms = sorted(
        scores.items(),
        key=lambda x: (-x[1], x[0]),
    )[:top_n]

    # Ensure JSON-serializable types only (str, int, float, bool, list, dict)
    keywords = [{"word": str(w), "score": round(float(s), 4)} for w, s in sorted_terms]
    result: dict[str, Any] = {
        "keywords": keywords,
        "metadata": {
            "total_terms": int(len(scores)),
            "top_n": int(top_n),
            "nltk_used": bool(NLTK_AVAILABLE),
        },
    }
    return result


def extract_keywords_json(
    text: str,
    top_n: int = 10,
    min_word_length: int = 2,
    use_stopwords: bool = True,
    indent: int | None = 2,
) -> str:
    """
    Extract keywords and return the result as a JSON string.

    This is a convenience wrapper around `extract_keywords()` for API responses,
    storage, or logging.

    Args:
        text: Same as `extract_keywords`.
        top_n: Same as `extract_keywords`.
        min_word_length: Same as `extract_keywords`.
        use_stopwords: Same as `extract_keywords`.
        indent: JSON indentation level. Use `None` for compact JSON.

    Returns:
        A JSON string containing the same structure returned by `extract_keywords`.

    Raises:
        TypeError, ValueError: Propagated from `extract_keywords` for invalid inputs.
    """
    data = extract_keywords(
        text=text,
        top_n=top_n,
        min_word_length=min_word_length,
        use_stopwords=use_stopwords,
    )
    return json.dumps(data, indent=indent)


class Tool:
    """
    Minimal wrapper class for AI agent tools.

    Many agent frameworks represent tools as a (name, description, callable) bundle.
    This wrapper follows the assignment specification and provides a single `execute`
    method that forwards keyword arguments to the underlying function.

    Attributes:
        name: Human-readable tool identifier.
        description: Short description of what the tool does, including expected inputs.
        fn: The callable to invoke when `execute` is called.
    """

    def __init__(self, name: str, description: str, fn):
        self.name = name
        self.description = description
        self.fn = fn

    def execute(self, **kwargs):
        """
        Execute the wrapped tool function.

        Args:
            **kwargs: Keyword arguments forwarded directly to the wrapped function.

        Returns:
            Whatever the wrapped function returns. For `keyword_extractor`, this is a
            JSON-serializable dict with `keywords` and `metadata`.

        Raises:
            Any exception raised by the wrapped function (e.g., `TypeError`, `ValueError`).
        """
        return self.fn(**kwargs)


# Tool instance for keyword extraction
keyword_extractor = Tool(
    name="keyword_extractor",
    description="Extracts the most important keywords from text using TF-IDF. "
    "Accepts text, top_n (number of keywords), and optional min_word_length, use_stopwords. "
    "Returns a JSON-serializable dict with 'keywords' and 'metadata'.",
    fn=extract_keywords,
)


if __name__ == "__main__":
    # Example usage and quick validation
    sample = (
        "Machine learning is a subset of artificial intelligence. "
        "Machine learning algorithms build models from sample data. "
        "Deep learning uses neural networks with many layers."
    )
    result = extract_keywords(sample, top_n=5)
    print(extract_keywords_json(sample, top_n=5))
    assert "keywords" in result and "metadata" in result
    assert len(result["keywords"]) <= 5
    assert json.dumps(result)  # ensure JSON-serializable
    print("OK: extract_keywords returns JSON-serializable dict.")

    # Tool wrapper example
    tool_result = keyword_extractor.execute(text=sample, top_n=5)
    print("Tool.execute result:", tool_result["keywords"][:3])
