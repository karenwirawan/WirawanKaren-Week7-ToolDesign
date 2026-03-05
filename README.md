# Keyword Extraction Tool

A keyword extraction tool for AI agents that identifies the most important terms in a text. It supports configurable output size, clear error handling, and returns JSON-serializable results for use in pipelines and APIs.

---

## What It Does

The tool takes raw text and returns a ranked list of keywords with TF-IDF–style scores. It is intended for **text analysis** tasks such as summarization, topic detection, tagging, and feeding downstream agents or search systems. The `Tool` wrapper (`keyword_extractor`) exposes this as a single call: `execute(text=..., top_n=...)`.

---

## Requirements

- **Python:** 3.7+
- **External packages:** Listed in `requirements.txt`. The only dependency is **NLTK** (optional but recommended for tokenization and stopword removal):

```bash
pip install -r requirements.txt
```

Without NLTK, the tool falls back to regex tokenization and skips stopword filtering.

---

## How to Run the Demo

Run the demo (successful runs, error handling including invalid types/values, and a simple agent workflow):

```bash
python demo.py
```

Use the tool in code:

```python
from tool import keyword_extractor

result = keyword_extractor.execute(text="Your document here.", top_n=10)
# result["keywords"] → [{"word": "...", "score": ...}, ...]
```

---

## Design Decisions

- **Why keyword extraction:** It reduces long text to a small set of representative terms, which helps with indexing, topic labels, and giving agents a compact view of content without reprocessing the full text.
- **Algorithm:** A **TF-IDF-style** score is used: the input is split into sentences (as “documents”), then term frequency and inverse document frequency are computed so that frequent, discriminative terms rank higher. **NLTK** is used when available for tokenization and English stopword removal; otherwise a regex-based tokenizer runs with no stopword list.

---

## Limitations

- **Language:** Optimized for English (NLTK stopwords and tokenizer). The fallback tokenizer is ASCII-oriented.
- **Single-document:** TF-IDF is computed only within the given text (sentences as documents), not against an external corpus, so IDF is relative to that text only.
- **No stemming/lemmatization:** Words like “running” and “run” are treated as different terms.
- **Without NLTK:** If NLTK is not installed, stopword filtering is skipped, so common words (e.g. “the”, “is”) can appear in the keyword list.
