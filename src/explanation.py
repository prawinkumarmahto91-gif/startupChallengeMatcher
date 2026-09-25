"""
src/explanation.py

Generates deterministic, human-readable explanations for why a startup
was recommended for a challenge. Every sentence is derived directly from
an actual computed feature value or raw data field - never invented.

Also assembles the final structured recommendation record matching the
project's spec output format (startup_id, scores, reasons, etc.).
"""

import json
from typing import List

import pandas as pd

from src.features import _split_technologies, GOV_EXPERIENCE_KEYWORDS


def explain_semantic(semantic_score: float) -> str:
    """Describe the overall semantic similarity level."""
    if semantic_score >= 0.75:
        return "Strong overall semantic similarity between the startup's profile and the challenge."
    elif semantic_score >= 0.5:
        return "Moderate semantic similarity between the startup's profile and the challenge."
    return "Limited semantic similarity between the startup's profile and the challenge."


def explain_technology(startup_row: pd.Series, challenge_row: pd.Series) -> str:
    """Describe technology overlap, naming the actual matched technologies."""
    startup_techs = set(_split_technologies(startup_row["technologies"]))
    challenge_techs = set(_split_technologies(challenge_row["technologies"]))
    matched = startup_techs & challenge_techs

    if not challenge_techs:
        return "No specific technology requirements were listed for this challenge."

    if matched and len(matched) == len(challenge_techs):
        matched_display = ", ".join(sorted(matched))
        return f"Strong match with all required technologies ({matched_display})."
    elif matched:
        matched_display = ", ".join(sorted(matched))
        return (f"Partial match with required technologies - covers {len(matched)} "
                f"of {len(challenge_techs)} required ({matched_display}).")
    return "No overlap found with the challenge's required technologies."


def explain_sector(startup_row: pd.Series, challenge_row: pd.Series) -> str:
    """Describe whether the startup's sector matches the challenge's sector."""
    if str(startup_row["sector"]).strip().lower() == str(challenge_row["sector"]).strip().lower():
        return f"Startup operates in the same sector as the challenge ({startup_row['sector']})."
    return (f"Startup's sector ({startup_row['sector']}) differs from the "
            f"challenge's sector ({challenge_row['sector']}).")


def explain_experience(startup_row: pd.Series) -> str:
    """Describe years of experience, flagging relevant government project history."""
    years = startup_row["experience"]
    previous_projects = str(startup_row.get("previous_projects", "")).lower()
    has_gov_experience = any(k in previous_projects for k in GOV_EXPERIENCE_KEYWORDS)

    base = f"{years} years of relevant experience"
    if has_gov_experience:
        return f"{base}, including prior government/municipal project experience ({startup_row['previous_projects']})."
    return f"{base}, with no prior government project experience noted."


def explain_budget(startup_row: pd.Series, challenge_row: pd.Series) -> str:
    """Describe whether the startup's cost fits within the challenge's budget."""
    startup_budget = float(startup_row["budget"])
    challenge_budget = float(challenge_row["budget"])

    if startup_budget <= challenge_budget:
        return f"Startup's estimated cost ({startup_budget}) is within the challenge's budget ({challenge_budget})."

    overage_pct = round(((startup_budget - challenge_budget) / challenge_budget) * 100, 1)
    return (f"Startup's estimated cost ({startup_budget}) exceeds the challenge's "
            f"budget ({challenge_budget}) by {overage_pct}%.")


def explain_location(startup_row: pd.Series, challenge_row: pd.Series) -> str:
    """Describe whether the startup is based in the challenge's deployment city."""
    if str(startup_row["location"]).strip().lower() == str(challenge_row["location"]).strip().lower():
        return f"Startup is located in the deployment city ({startup_row['location']})."
    return (f"Startup is based in {startup_row['location']}, "
            f"not the deployment city ({challenge_row['location']}).")


def generate_explanation(startup_row: pd.Series, challenge_row: pd.Series) -> List[str]:
    """Generate the full list of deterministic reasons for one recommendation."""
    return [
        explain_semantic(startup_row["semantic_score"]),
        explain_technology(startup_row, challenge_row),
        explain_sector(startup_row, challenge_row),
        explain_experience(startup_row),
        explain_budget(startup_row, challenge_row),
        explain_location(startup_row, challenge_row),
    ]


def build_recommendation_record(startup_row: pd.Series, challenge_row: pd.Series) -> dict:
    """Build the final structured recommendation record for one startup,
    matching the project spec's output format.
    """
    return {
        "startup_id": int(startup_row["startup_id"]),
        "startup_name": startup_row["name"],
        "rank": int(startup_row["rank"]),
        "final_score": float(startup_row["final_score"]),
        "semantic_score": round(float(startup_row["semantic_score"]), 4),
        "technology_match": float(startup_row["technology_match"]),
        "sector_match": float(startup_row["sector_match"]),
        "experience_score": float(startup_row["experience_score"]),
        "budget_score": float(startup_row["budget_score"]),
        "location_score": float(startup_row["location_score"]),
        "reasons": generate_explanation(startup_row, challenge_row),
    }


if __name__ == "__main__":
    from src.ranking import rank_candidates
    from src.similarity import load_ready_data

    test_challenge_id = 102
    _, challenges = load_ready_data()
    challenge_row = challenges[challenges["challenge_id"] == test_challenge_id].iloc[0]

    ranked = rank_candidates(test_challenge_id, top_k_semantic=20, top_n_final=10)

    records = [build_recommendation_record(row, challenge_row) for _, row in ranked.iterrows()]

    print(f"Challenge: {challenge_row['title']}\n")
    print("Top recommendation (full structured record):")
    print(json.dumps(records[0], indent=2))

    print("\nReasons for all ranked candidates:")
    for record in records:
        print(f"\nRank {record['rank']}: {record['startup_name']} (score: {record['final_score']})")
        for reason in record["reasons"]:
            print(f"  - {reason}")