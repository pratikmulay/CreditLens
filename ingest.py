"""
ingest.py  –  Load raw_credit_data.csv into ChromaDB collection 'sme_profiles'

Each CSV row is mapped to an SME-style document:
  - document text: human-readable summary sentence
  - metadata: all numeric/categorical fields + derived risk_label
  - id: "record_<row_index>"

Progress is printed every 100 records.
"""

import os
import csv
import chromadb
from chromadb.config import Settings

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "data", "raw_credit_data.csv")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

BATCH_SIZE = 100          # rows ingested per ChromaDB upsert call
PROGRESS_EVERY = 100     # print progress every N records


def risk_label(loan_status: str) -> str:
    """Map loan_status (0 = repaid, 1 = defaulted) to a human label."""
    try:
        return "high" if int(float(loan_status)) == 1 else "low"
    except (ValueError, TypeError):
        return "unknown"


def safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val: str, default: int = 0) -> int:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def row_to_document(row: dict, idx: int) -> str:
    """Build a natural-language description for embedding."""
    intent = row.get("loan_intent", "unknown").lower().replace("_", " ")
    ownership = row.get("person_home_ownership", "unknown").lower()
    grade = row.get("loan_grade", "?")
    income = safe_int(row.get("person_income", "0"))
    loan_amt = safe_int(row.get("loan_amnt", "0"))
    int_rate = safe_float(row.get("loan_int_rate", "0"))
    emp_len = safe_float(row.get("person_emp_length", "0"))
    cred_hist = safe_int(row.get("cb_person_cred_hist_length", "0"))
    default_flag = row.get("cb_person_default_on_file", "N")
    pct_income = safe_float(row.get("loan_percent_income", "0"))
    risk = risk_label(row.get("loan_status", "0"))

    return (
        f"SME applicant with annual income ${income:,}, seeking ${loan_amt:,} loan "
        f"for {intent} purposes (grade {grade}, {int_rate:.1f}% interest). "
        f"Home ownership: {ownership}. Employment: {emp_len:.0f} years. "
        f"Credit history: {cred_hist} years, prior default on file: {default_flag}. "
        f"Loan-to-income ratio: {pct_income:.0%}. Risk profile: {risk}."
    )


def row_to_metadata(row: dict) -> dict:
    """Return a flat metadata dict (ChromaDB requires str/int/float/bool values)."""
    return {
        "person_age":                safe_int(row.get("person_age")),
        "person_income":             safe_int(row.get("person_income")),
        "person_home_ownership":     str(row.get("person_home_ownership", "")),
        "person_emp_length":         safe_float(row.get("person_emp_length")),
        "loan_intent":               str(row.get("loan_intent", "")),
        "loan_grade":                str(row.get("loan_grade", "")),
        "loan_amnt":                 safe_int(row.get("loan_amnt")),
        "loan_int_rate":             safe_float(row.get("loan_int_rate")),
        "loan_status":               safe_int(row.get("loan_status")),
        "loan_percent_income":       safe_float(row.get("loan_percent_income")),
        "cb_person_default_on_file": str(row.get("cb_person_default_on_file", "N")),
        "cb_person_cred_hist_length":safe_int(row.get("cb_person_cred_hist_length")),
        "risk_label":                risk_label(row.get("loan_status", "0")),
        "source":                    "raw_csv",
    }


def main():
    # ── connect to ChromaDB ──────────────────────────────────────────────────
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection so we start fresh each run
    try:
        client.delete_collection("sme_profiles")
        print("Existing 'sme_profiles' collection cleared.")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="sme_profiles",
        metadata={"hnsw:space": "cosine"},
    )

    # ── read CSV and ingest in batches ───────────────────────────────────────
    docs, metas, ids = [], [], []
    total = 0

    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            doc_id = f"record_{i}"
            docs.append(row_to_document(row, i))
            metas.append(row_to_metadata(row))
            ids.append(doc_id)
            total += 1

            # flush batch
            if len(docs) >= BATCH_SIZE:
                collection.upsert(documents=docs, metadatas=metas, ids=ids)
                docs, metas, ids = [], [], []

            if total % PROGRESS_EVERY == 0:
                print(f"Progress: {total} records ingested...")

    # flush remainder
    if docs:
        collection.upsert(documents=docs, metadatas=metas, ids=ids)

    print(f"\nTotal documents ingested: {collection.count()}")


if __name__ == "__main__":
    main()
