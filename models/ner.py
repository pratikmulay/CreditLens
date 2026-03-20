"""
models/ner.py  –  Financial Named-Entity Recognition.

Strategy
--------
• HuggingFace NER pipeline (dslim/bert-base-NER) detects standard entity
  types: ORG, PER, LOC, MISC.
• A set of regex patterns extracts monetary / revenue figures that BERT-NER
  misses (e.g. "$2.5 million", "₹2 crore", "INR 5 lakh").

Returned dict
-------------
{
  "org_mentions":     ["Acme Corp", ...],   # de-duplicated ORG entities
  "person_mentions":  ["John Smith", ...],  # de-duplicated PER entities
  "revenue_mentions": ["2.5 million", ...], # regex-extracted monetary values
  "all_entities":     [...],                # raw entity list from BERT-NER
}
"""

from __future__ import annotations
import re
from transformers import pipeline, Pipeline

_NER_PIPELINE: Pipeline | None = None
_MODEL_NAME = "dslim/bert-base-NER"

# ── regex for monetary / revenue figures ─────────────────────────────────────
_MONEY_PATTERNS = [
    # "$2.5 million"  "$500,000"  "$1.2B"
    r"\$\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand|lakh|crore|[BbMmKk])\b",
    r"\$\s*[\d,]+(?:\.\d+)?",
    # "₹ 2 crore"  "INR 5 lakh"  "Rs. 10 crore"
    r"(?:₹|INR|Rs\.?|Rs)\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand|lakh|crore|[BbMmKk])\b",
    r"(?:₹|INR|Rs\.?|Rs)\s*[\d,]+(?:\.\d+)?",
    # "2.5 million"  "10 crore"  "50 lakh" standalone
    r"\b[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand|lakh|crore)\b",
    # "revenue of X"  / "turnover of X"
    r"(?:revenue|turnover|sales|income)\s+(?:of\s+)?(?:approximately\s+)?(?:[\$₹INRRs\.]*\s*)[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand|lakh|crore|[BbMmKk])?\b",
]

_MONEY_RE = re.compile("|".join(_MONEY_PATTERNS), re.IGNORECASE)


def _get_ner_pipeline() -> Pipeline:
    global _NER_PIPELINE
    if _NER_PIPELINE is None:
        _NER_PIPELINE = pipeline(
            "ner",
            model=_MODEL_NAME,
            tokenizer=_MODEL_NAME,
            aggregation_strategy="simple",   # merge sub-tokens into whole words
            device=-1,                       # CPU
        )
    return _NER_PIPELINE


def _extract_revenue_mentions(text: str) -> list[str]:
    """Return de-duplicated list of monetary value strings found in *text*."""
    raw_matches = _MONEY_RE.findall(text)
    # findall with alternation returns tuples; collapse to first non-empty group
    cleaned: list[str] = []
    for m in raw_matches:
        val = m if isinstance(m, str) else next((x for x in m if x), "")
        val = re.sub(r"\s+", " ", val).strip()
        if val and val not in cleaned:
            cleaned.append(val)
    return cleaned


def _merge_entities(ner_results: list[dict]) -> dict[str, list[str]]:
    """Group NER entities by type and de-duplicate."""
    org_mentions:    list[str] = []
    person_mentions: list[str] = []

    for ent in ner_results:
        word = ent.get("word", "").strip()
        etype = ent.get("entity_group", "")
        if not word:
            continue
        if etype == "ORG" and word not in org_mentions:
            org_mentions.append(word)
        elif etype == "PER" and word not in person_mentions:
            person_mentions.append(word)

    return {"org_mentions": org_mentions, "person_mentions": person_mentions}


class FinancialNER:
    """
    Financial Named-Entity Recognition model.

    Usage
    -----
    model = FinancialNER()
    result = model.extract(
        'Acme Corp reported revenue of $2.5 million in Q3 2023, led by CEO John Smith.'
    )
    # → {
    #     "org_mentions":     ["Acme Corp"],
    #     "person_mentions":  ["John Smith"],
    #     "revenue_mentions": ["$2.5 million"],
    #     "all_entities":     [...],
    #   }
    """

    def __init__(self) -> None:
        self._pipe: Pipeline | None = None

    def _load(self) -> Pipeline:
        if self._pipe is None:
            print(f"Loading NER model from '{_MODEL_NAME}' …")
            self._pipe = _get_ner_pipeline()
        return self._pipe

    def extract(self, text: str) -> dict:
        """
        Extract named entities and revenue figures from *text*.

        Returns
        -------
        dict with keys: org_mentions, person_mentions, revenue_mentions, all_entities
        """
        pipe = self._load()

        # BERT-NER
        ner_results = pipe(text[:512])
        grouped = _merge_entities(ner_results)

        # Regex revenue extraction (run on full text, not truncated)
        revenue = _extract_revenue_mentions(text)

        return {
            "org_mentions":     grouped["org_mentions"],
            "person_mentions":  grouped["person_mentions"],
            "revenue_mentions": revenue,
            "all_entities":     [
                {
                    "entity_group": e.get("entity_group"),
                    "word":         e.get("word"),
                    "score":        round(float(e.get("score", 0)), 4),
                }
                for e in ner_results
            ],
        }
