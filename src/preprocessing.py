"""
src/preprocessing.py

Cleans raw startup and challenge data before it is used anywhere else
in the pipeline (text building, embeddings, feature engineering).

Responsibilities:
- Handle missing values (e.g. the "None" -> NaN certification issue)
- Remove duplicate rows
- Normalize whitespace in text fields
- Normalize sector names (so "Smart Cities" and "Smart City" match)
- Normalize technology names (so small spelling/casing differences
  don't break exact-match comparisons later)
"""

import re
import pandas as pd


# ---------------------------------------------------------------------------
# Synonym maps
#
# These map "messy" real-world variants to one canonical form. Our current
# synthetic data barely needs this, but this is the mechanism we will rely
# on once real / messier data arrives. Add more entries here over time.
# ---------------------------------------------------------------------------

SECTOR_SYNONYMS = {
    "smart cities": "Smart City",
    "smart-city": "Smart City",
    "waste mgmt": "Waste Management",
    "health": "Healthcare",
    "agri": "Agriculture",
}

TECH_SYNONYMS = {
    "artificial intelligence": "AI",
    "gps": "GPS tracking",
    "iot": "IoT sensors",
    "cv": "Computer vision",
}


def clean_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into one space and strip edges.

    Example: "  Pune   Municipal " -> "Pune Municipal"
    """
    if pd.isna(text):
        return text
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_certifications(value: str) -> str:
    """Turn missing or literal 'none' values into an explicit, safe string.

    Why: pandas reads the literal text "None" in a CSV as a missing value
    (NaN). We want a real string here so downstream code (e.g. checking
    "does the startup have ISO 27001?") never has to special-case NaN.
    """
    if pd.isna(value) or str(value).strip().lower() in ("none", ""):
        return "No certification"
    return clean_whitespace(value)


def normalize_sector(value: str) -> str:
    """Standardize sector text using the synonym map, else clean + keep as-is."""
    cleaned = clean_whitespace(value)
    if pd.isna(cleaned):
        return cleaned
    key = cleaned.lower()
    return SECTOR_SYNONYMS.get(key, cleaned)


def normalize_technologies(value: str) -> str:
    """Clean and standardize a comma-separated technology string.

    Steps:
    1. Split on commas
    2. Strip whitespace from each technology
    3. Map known synonyms to their canonical form
    4. Remove duplicates while preserving order
    5. Re-join into a single comma-separated string
    """
    if pd.isna(value):
        return value

    raw_items = [item.strip() for item in str(value).split(",") if item.strip()]

    normalized_items = []
    seen = set()
    for item in raw_items:
        canonical = TECH_SYNONYMS.get(item.lower(), item)
        if canonical.lower() not in seen:
            seen.add(canonical.lower())
            normalized_items.append(canonical)

    return ", ".join(normalized_items)


def remove_duplicate_rows(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Drop duplicate rows based on the unique ID column, keeping the first."""
    before = len(df)
    df = df.drop_duplicates(subset=id_column, keep="first").reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Removed {before - after} duplicate rows based on '{id_column}'.")
    return df


def preprocess_startups(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for the startups dataframe."""
    df = df.copy()

    df = remove_duplicate_rows(df, id_column="startup_id")

    text_columns = ["name", "description", "capabilities", "previous_projects", "location"]
    for col in text_columns:
        df[col] = df[col].apply(clean_whitespace)

    df["sector"] = df["sector"].apply(normalize_sector)
    df["technologies"] = df["technologies"].apply(normalize_technologies)
    df["certifications"] = df["certifications"].apply(normalize_certifications)

    return df


def preprocess_challenges(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for the challenges dataframe."""
    df = df.copy()

    df = remove_duplicate_rows(df, id_column="challenge_id")

    text_columns = ["title", "problem", "desired_outcome", "location", "mandatory_requirements"]
    for col in text_columns:
        df[col] = df[col].apply(clean_whitespace)

    df["sector"] = df["sector"].apply(normalize_sector)
    df["technologies"] = df["technologies"].apply(normalize_technologies)

    return df


if __name__ == "__main__":
    # Quick manual run: load raw data, clean it, show before/after, save results.
    raw_startups = pd.read_csv("data/startups.csv")
    raw_challenges = pd.read_csv("data/challenges.csv")

    print("Missing certifications before cleaning:",
          raw_startups["certifications"].isna().sum())

    clean_startups = preprocess_startups(raw_startups)
    clean_challenges = preprocess_challenges(raw_challenges)

    print("Missing certifications after cleaning:",
          clean_startups["certifications"].isna().sum())

    print("\nSample cleaned startup row:")
    print(clean_startups.loc[2, ["name", "sector", "technologies", "certifications"]])

    clean_startups.to_csv("data/startups_clean.csv", index=False)
    clean_challenges.to_csv("data/challenges_clean.csv", index=False)

    print("\nSaved data/startups_clean.csv and data/challenges_clean.csv")