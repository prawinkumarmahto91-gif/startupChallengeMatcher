"""
src/features.py

Computes individual ranking features comparing a startup to a challenge:
technology_match, sector_match, experience_score, budget_score, location_score.

Each function returns a float between 0 and 1, so all features share the
same scale before being combined in Day 10's hybrid ranking formula.
These features also power Day 11's explainability - each one maps
directly to a human-readable reason.
"""

from typing import List

import pandas as pd

GOV_EXPERIENCE_KEYWORDS = [
    "municipal", "government", "corporation", "mission",
    "department", "police", "state health",
]


def _split_technologies(tech_string: str) -> List[str]:
    """Split a comma-separated technology string into a clean lowercase list."""
    if pd.isna(tech_string):
        return []
    return [t.strip().lower() for t in str(tech_string).split(",") if t.strip()]


def technology_match(startup_technologies: str, challenge_technologies: str) -> float:
    """Fraction of the challenge's REQUIRED technologies that the startup covers.

    Example: challenge requires {A, B, C}, startup has {A, B, D} -> 2/3 = 0.67

    We measure against the challenge's requirement list (not a symmetric
    overlap) because what matters is how much of what the CHALLENGE needs
    is covered - extra unrelated tech on the startup's side shouldn't
    count against it.
    """
    startup_set = set(_split_technologies(startup_technologies))
    challenge_set = set(_split_technologies(challenge_technologies))

    if not challenge_set:
        return 0.0

    matched = startup_set & challenge_set
    return round(len(matched) / len(challenge_set), 4)


def sector_match(startup_sector: str, challenge_sector: str) -> float:
    """1.0 if sectors match exactly (case-insensitive), else 0.0."""
    if pd.isna(startup_sector) or pd.isna(challenge_sector):
        return 0.0
    return 1.0 if str(startup_sector).strip().lower() == str(challenge_sector).strip().lower() else 0.0


def experience_score(startup_row: pd.Series) -> float:
    """Score based on years of experience, with a bonus for relevant
    government/municipal project experience mentioned in previous_projects.
    """
    years = startup_row.get("experience", 0)
    base_score = min(years / 10.0, 1.0)  # 10+ years reaches the max score

    previous_projects = str(startup_row.get("previous_projects", "")).lower()
    has_gov_experience = any(keyword in previous_projects for keyword in GOV_EXPERIENCE_KEYWORDS)

    bonus = 0.1 if has_gov_experience else 0.0
    return round(min(base_score + bonus, 1.0), 4)


def budget_score(startup_budget: float, challenge_budget: float) -> float:
    """1.0 if the startup's cost fits within the challenge's budget.
    If over budget, the score decreases proportionally to how far over it is.
    """
    # Defensive type coercion: PostgreSQL NUMERIC columns come back as
    # decimal.Decimal (not float), which cannot be mixed with float in
    # arithmetic. Casting both to float here keeps this function correct
    # regardless of whether the data came from a CSV (already float) or
    # PostgreSQL (Decimal).
    startup_budget = float(startup_budget)
    challenge_budget = float(challenge_budget)

    if challenge_budget <= 0:
        return 0.0

    if startup_budget <= challenge_budget:
        return 1.0

    overage_ratio = (startup_budget - challenge_budget) / challenge_budget
    return round(max(0.0, 1.0 - overage_ratio), 4)


def location_score(startup_location: str, challenge_location: str) -> float:
    """1.0 if the startup is based in the same city as the challenge's
    deployment location, otherwise a lower baseline score (deployment is
    still possible, just without the local-presence advantage).
    """
    if pd.isna(startup_location) or pd.isna(challenge_location):
        return 0.0
    same_city = str(startup_location).strip().lower() == str(challenge_location).strip().lower()
    return 1.0 if same_city else 0.4


def compute_features(startup_row: pd.Series, challenge_row: pd.Series) -> dict:
    """Compute all five ranking features for one startup-challenge pair."""
    return {
        "technology_match": technology_match(startup_row["technologies"], challenge_row["technologies"]),
        "sector_match": sector_match(startup_row["sector"], challenge_row["sector"]),
        "experience_score": experience_score(startup_row),
        "budget_score": budget_score(startup_row["budget"], challenge_row["budget"]),
        "location_score": location_score(startup_row["location"], challenge_row["location"]),
    }


def add_features_to_candidates(challenge_row: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach all five features to every row in candidates.

    candidates must include these startup fields: technologies, sector,
    experience, previous_projects, budget, location.
    """
    result = candidates.copy()

    feature_rows = [compute_features(row, challenge_row) for _, row in candidates.iterrows()]
    features_df = pd.DataFrame(feature_rows, index=candidates.index)

    return pd.concat([result, features_df], axis=1)


if __name__ == "__main__":
    from src.similarity import get_top_candidates, load_ready_data
    from src.eligibility import filter_eligible_startups

    startups, challenges = load_ready_data()

    test_challenge_id = 102
    challenge_row = challenges[challenges["challenge_id"] == test_challenge_id].iloc[0]
    print(f"Challenge: {challenge_row['title']}")
    print(f"Required technologies: {challenge_row['technologies']}")
    print(f"Budget: {challenge_row['budget']} | Location: {challenge_row['location']}\n")

    top_candidates = get_top_candidates(challenge_id=test_challenge_id, top_k=20)

    # Merge in the extra startup fields needed for eligibility + features.
    # NOTE: 'sector' is already present in top_candidates, so we don't
    # merge it again here (that would create duplicate columns).
    extra_cols = ["startup_id", "technologies", "experience", "previous_projects",
                  "budget", "location", "certifications", "dpiit_recognized"]
    candidates_full = top_candidates.merge(startups[extra_cols], on="startup_id")

    checked = filter_eligible_startups(challenge_row, candidates_full)
    eligible = checked[checked["eligible"]].copy()
    print(f"{len(eligible)} eligible candidates out of {len(checked)}.\n")

    with_features = add_features_to_candidates(challenge_row, eligible)

    display_cols = ["startup_id", "name", "semantic_score", "technology_match",
                     "sector_match", "experience_score", "budget_score",
                     "location_score", "budget", "location"]
    print(with_features[display_cols].to_string(index=False))