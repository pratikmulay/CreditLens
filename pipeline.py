"""
pipeline.py  –  CreditLens full-stack analysis pipeline.

Orchestrates:
  1. FinBERT sentiment  (models/sentiment.py)
  2. Financial NER      (models/ner.py)
  3. Semantic search    (models/embedder.py  →  ChromaDB)
  4. Groq LLM           (generates structured credit brief)
  5. News fetch         (optional, uses NEWS_API_KEY from .env)

Entry point
-----------
pipeline = CreditLensPipeline()
result = pipeline.analyze(
    business_name="Sharma Textiles",
    description="...",
    financial_text="...",
    fetch_news=False,
)
# result["credit_brief"] has keys: risk_rating, risk_score, recommendation
"""

from __future__ import annotations
import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from models.sentiment import FinBERTSentiment
from models.ner import FinancialNER
from models.embedder import ProfileEmbedder

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_ENV_PATH, override=True)

_GROQ_MODEL = "llama-3.1-8b-instant"   # updated fast model on free tier


# ── helpers ───────────────────────────────────────────────────────────────────

def _fetch_news(business_name: str, api_key: str, max_articles: int = 3) -> list[dict]:
    """Fetch recent news about *business_name* via NewsAPI."""
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": business_name,
            "sortBy": "publishedAt",
            "pageSize": max_articles,
            "language": "en",
            "apiKey": api_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [
            {
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "source":      a.get("source", {}).get("name", ""),
                "published":   a.get("publishedAt", ""),
            }
            for a in articles
        ]
    except Exception as exc:
        return [{"error": str(exc)}]


def _build_groq_prompt(
    business_name: str,
    description: str,
    financial_text: str,
    sentiment_result: dict,
    ner_result: dict,
    similar_profiles: list[dict],
    news_articles: list[dict],
) -> str:
    """Construct the LLM prompt with all available context."""
    # Summarise similar profiles
    profile_summary = ""
    for i, p in enumerate(similar_profiles[:3], 1):
        profile_summary += (
            f"  {i}. Grade {p.get('loan_grade','?')}, "
            f"income ${p.get('person_income', 0):,}, "
            f"loan ${p.get('loan_amnt', 0):,}, "
            f"risk: {p.get('risk_label', '?')}, "
            f"default history: {p.get('cb_person_default_on_file', 'N')}\n"
        )

    news_summary = ""
    for a in news_articles[:3]:
        if "error" not in a:
            news_summary += f"  - [{a['source']}] {a['title']}\n"
    if not news_summary:
        news_summary = "  (no recent news fetched)\n"

    org_str = ", ".join(ner_result.get("org_mentions", [])) or "none detected"
    rev_str = ", ".join(ner_result.get("revenue_mentions", [])) or "not specified"

    prompt = f"""You are CreditLens, an expert SME credit risk analyst.

## Business Profile
Name        : {business_name}
Description : {description}
Financial   : {financial_text}

## AI Analysis Results
Sentiment   : {sentiment_result.get('label', 'N/A')} (confidence {sentiment_result.get('score', 0):.2%})
Orgs found  : {org_str}
Revenue mentions: {rev_str}

## Similar Historical Cases (from credit database)
{profile_summary or "  (no similar cases found)"}

## Recent News
{news_summary}

## Your Task
Produce a concise credit risk brief in **valid JSON** with EXACTLY these fields:
{{
  "risk_rating":     "Low" | "Medium" | "High",
  "risk_score":      <integer 1-100, higher = riskier>,
  "recommendation":  "Approve" | "Approve with conditions" | "Reject",
  "summary":         "<2-3 sentence plain-English summary>",
  "key_risks":       ["<risk 1>", "<risk 2>", "<risk 3>"],
  "mitigants":       ["<mitigant 1>", "<mitigant 2>"],
  "suggested_loan_grade": "A" | "B" | "C" | "D" | "E" | "F"
}}

Return ONLY the JSON object. No markdown, no explanation outside the JSON.
"""
    return prompt


def _parse_credit_brief(raw: str) -> dict:
    """Robustly extract the JSON credit brief from the LLM response."""
    raw = raw.strip()

    # If LLM wrapped in markdown fences, strip them
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(
            l for l in lines
            if not l.strip().startswith("```")
        )

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Find outermost { … } block
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Fallback: return raw text wrapped in a dict
    return {
        "risk_rating":          "Unknown",
        "risk_score":           50,
        "recommendation":       "Manual review required",
        "summary":              raw[:500],
        "key_risks":            [],
        "mitigants":            [],
        "suggested_loan_grade": "C",
    }


# ── main pipeline ─────────────────────────────────────────────────────────────

class CreditLensPipeline:
    """
    Full CreditLens analysis pipeline.

    Lazy-loads all sub-models on first call to analyze().
    """

    def __init__(self) -> None:
        self._sentiment: FinBERTSentiment | None = None
        self._ner:       FinancialNER | None = None
        self._embedder:  ProfileEmbedder | None = None
        self._groq:      Groq | None = None

        self._groq_key    = os.getenv("GROQ_API_KEY", "")
        self._news_key    = os.getenv("NEWS_API_KEY", "")

    # ── lazy loaders ──────────────────────────────────────────────────────────
    def _get_sentiment(self) -> FinBERTSentiment:
        if self._sentiment is None:
            self._sentiment = FinBERTSentiment()
        return self._sentiment

    def _get_ner(self) -> FinancialNER:
        if self._ner is None:
            self._ner = FinancialNER()
        return self._ner

    def _get_embedder(self) -> ProfileEmbedder:
        if self._embedder is None:
            self._embedder = ProfileEmbedder()
        return self._embedder

    def _get_groq(self) -> Groq:
        if self._groq is None:
            if not self._groq_key:
                raise ValueError("GROQ_API_KEY not set in environment / .env file")
            self._groq = Groq(api_key=self._groq_key)
        return self._groq

    # ── public API ────────────────────────────────────────────────────────────
    def analyze(
        self,
        business_name: str,
        description: str,
        financial_text: str,
        fetch_news: bool = True,
        n_similar: int = 5,
    ) -> dict:
        """
        Run the full CreditLens pipeline for an SME.

        Parameters
        ----------
        business_name  – display name of the business
        description    – qualitative profile (sector, size, years, employees …)
        financial_text – any financial information available (revenue, ratios …)
        fetch_news     – if True, calls NewsAPI; set False during testing
        n_similar      – how many similar historical cases to retrieve

        Returns
        -------
        dict with keys:
          business_name, timestamp, sentiment, ner, similar_profiles,
          news_articles, credit_brief (contains risk_rating, risk_score,
          recommendation, summary, key_risks, mitigants, suggested_loan_grade)
        """
        timestamp = datetime.now().isoformat()
        combined_text = f"{description}. {financial_text}"

        # ── Step 1: Sentiment ─────────────────────────────────────────────────
        print("→ Running FinBERT sentiment …")
        sentiment_result = self._get_sentiment().analyze(financial_text)

        # ── Step 2: NER ───────────────────────────────────────────────────────
        print("→ Running financial NER …")
        ner_result = self._get_ner().extract(combined_text)

        # ── Step 3: Semantic search ───────────────────────────────────────────
        print("→ Searching similar profiles in ChromaDB …")
        similar_profiles = self._get_embedder().find_similar(
            combined_text, n=n_similar
        )

        # ── Step 4: News (optional) ───────────────────────────────────────────
        news_articles: list[dict] = []
        if fetch_news and self._news_key:
            print("→ Fetching news articles …")
            news_articles = _fetch_news(business_name, self._news_key)
        else:
            print("→ Skipping news fetch.")

        # ── Step 5: Groq LLM credit brief ────────────────────────────────────
        print("→ Generating credit brief via Groq LLM …")
        prompt = _build_groq_prompt(
            business_name=business_name,
            description=description,
            financial_text=financial_text,
            sentiment_result=sentiment_result,
            ner_result=ner_result,
            similar_profiles=similar_profiles,
            news_articles=news_articles,
        )

        groq_client = self._get_groq()
        chat_resp = groq_client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )
        raw_output = chat_resp.choices[0].message.content
        credit_brief = _parse_credit_brief(raw_output)

        return {
            "business_name":   business_name,
            "timestamp":       timestamp,
            "sentiment":       sentiment_result,
            "ner":             ner_result,
            "similar_profiles": similar_profiles,
            "news_articles":   news_articles,
            "credit_brief":    credit_brief,
        }
