"""
models/sentiment.py  –  FinBERT-based financial sentiment analyser.

Uses ProsusAI/finbert (fine-tuned BERT for financial text).
Returns label ∈ {positive, negative, neutral} and a confidence float.
Model is downloaded once and cached by HuggingFace.
"""

from __future__ import annotations
from transformers import pipeline, Pipeline

_PIPELINE: Pipeline | None = None
_MODEL_NAME = "ProsusAI/finbert"


def _get_pipeline() -> Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = pipeline(
            "text-classification",
            model=_MODEL_NAME,
            tokenizer=_MODEL_NAME,
            top_k=1,          # return only best label
            truncation=True,
            max_length=512,
        )
    return _PIPELINE


class FinBERTSentiment:
    """
    Lazy-loaded FinBERT sentiment model.

    Usage
    -----
    model = FinBERTSentiment()
    result = model.analyze("Strong revenue growth this quarter.")
    # → {"label": "positive", "score": 0.987}
    """

    def __init__(self) -> None:
        self._pipe: Pipeline | None = None

    def _load(self) -> Pipeline:
        if self._pipe is None:
            print(f"Loading FinBERT from '{_MODEL_NAME}' …")
            self._pipe = _get_pipeline()
        return self._pipe

    def analyze(self, text: str) -> dict:
        """
        Analyse financial sentiment of *text*.

        Returns
        -------
        dict with keys:
          label  – "positive" | "negative" | "neutral"
          score  – confidence float in [0, 1]
          text   – first 200 chars of the input (for debugging)
        """
        pipe = self._load()

        # pipeline returns [[{label, score}]] when top_k=1
        raw = pipe(text[:512])
        best = raw[0][0] if isinstance(raw[0], list) else raw[0]

        label = best["label"].lower()      # finbert outputs Positive/Negative/Neutral
        score = round(float(best["score"]), 4)

        return {
            "label": label,
            "score": score,
            "text_preview": text[:200],
        }

    def batch_analyze(self, texts: list[str]) -> list[dict]:
        """Analyse multiple texts in one forward pass."""
        pipe = self._load()
        raw_list = pipe([t[:512] for t in texts])
        results = []
        for raw, text in zip(raw_list, texts):
            best = raw[0] if isinstance(raw, list) else raw
            results.append({
                "label": best["label"].lower(),
                "score": round(float(best["score"]), 4),
                "text_preview": text[:200],
            })
        return results
