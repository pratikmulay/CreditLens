"""
generate_synthetic.py  –  Generate 50 synthetic SME credit profiles and
                          upsert them into the existing ChromaDB 'sme_profiles'
                          collection.

Each synthetic profile is crafted with realistic variance so it is meaningfully
different from the CSV-derived records, while still carrying the same fields
(and in particular a 'risk_label' in metadata).
"""

import os
import random
import chromadb

# ── reproducibility ──────────────────────────────────────────────────────────
random.seed(42)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
N_PROFILES = 50

# ── lookup tables ─────────────────────────────────────────────────────────────
BUSINESS_TYPES = [
    "small retail business",
    "mid-size manufacturing firm",
    "tech startup",
    "agricultural cooperative",
    "logistics company",
    "food & beverage SME",
    "construction contractor",
    "healthcare clinic",
    "e-commerce store",
    "professional services firm",
]

LOAN_INTENTS = ["VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT",
                "PERSONAL", "MEDICAL", "EDUCATION"]

LOAN_GRADES = ["A", "B", "C", "D", "E", "F"]

GRADE_RATE_RANGE = {
    "A": (5.5,  9.0),
    "B": (9.0,  12.0),
    "C": (12.0, 14.5),
    "D": (14.5, 17.0),
    "E": (17.0, 19.5),
    "F": (19.5, 22.0),
}

REVENUE_DESCRIPTORS = ["low", "moderate", "above-average", "high", "very high"]
HOME_OWNERSHIP = ["RENT", "OWN", "MORTGAGE"]


def generate_profile(idx: int) -> dict:
    """Return a dict with document text, metadata, and a unique id."""
    biz_type   = random.choice(BUSINESS_TYPES)
    intent     = random.choice(LOAN_INTENTS)
    grade      = random.choice(LOAN_GRADES)
    income     = random.randint(18_000, 400_000)
    loan_amt   = random.randint(1_000, 35_000)
    int_rate   = round(random.uniform(*GRADE_RATE_RANGE[grade]), 2)
    emp_length = round(random.uniform(0, 15), 1)
    cred_hist  = random.randint(2, 30)
    default_flag = random.choices(["Y", "N"], weights=[15, 85])[0]
    ownership  = random.choice(HOME_OWNERSHIP)
    age        = random.randint(21, 65)
    pct_income = round(loan_amt / income, 2)

    # risk heuristic: grade D-F or prior default → likely high risk
    if grade in ("D", "E", "F") or default_flag == "Y":
        risk = "high" if random.random() < 0.75 else "low"
        loan_status = 1 if risk == "high" else 0
    else:
        risk = "low" if random.random() < 0.80 else "high"
        loan_status = 0 if risk == "low" else 1

    revenue_desc = random.choice(REVENUE_DESCRIPTORS)
    intent_clean = intent.lower().replace("_", " ")

    document = (
        f"Synthetic SME: {biz_type} with {revenue_desc} revenue (~${income:,}/yr), "
        f"requesting ${loan_amt:,} loan for {intent_clean} (grade {grade}, "
        f"{int_rate:.1f}% interest). Owner age {age}, employment {emp_length:.0f} yrs, "
        f"home: {ownership.lower()}. Credit history {cred_hist} yrs, "
        f"prior default: {default_flag}. Loan-to-income: {pct_income:.0%}. "
        f"Assessed risk: {risk}."
    )

    metadata = {
        "person_age":                age,
        "person_income":             income,
        "person_home_ownership":     ownership,
        "person_emp_length":         emp_length,
        "loan_intent":               intent,
        "loan_grade":                grade,
        "loan_amnt":                 loan_amt,
        "loan_int_rate":             int_rate,
        "loan_status":               loan_status,
        "loan_percent_income":       pct_income,
        "cb_person_default_on_file": default_flag,
        "cb_person_cred_hist_length":cred_hist,
        "risk_label":                risk,
        "source":                    "synthetic",
        "business_type":             biz_type,
        "revenue_descriptor":        revenue_desc,
    }

    return {
        "id":       f"synthetic_{idx}",
        "document": document,
        "metadata": metadata,
    }


def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name="sme_profiles",
        metadata={"hnsw:space": "cosine"},
    )

    profiles = [generate_profile(i) for i in range(N_PROFILES)]

    collection.upsert(
        documents=[p["document"] for p in profiles],
        metadatas=[p["metadata"] for p in profiles],
        ids=[p["id"] for p in profiles],
    )

    print(f"Generated {N_PROFILES} synthetic profiles")
    print(f"Ingested into ChromaDB. Total collection size: {collection.count()}")


if __name__ == "__main__":
    main()
