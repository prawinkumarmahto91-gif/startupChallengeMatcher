"""
src/eligibility.py

Applies HARD eligibility rules to startups before any ranking happens.

Rule from the project spec: eligibility is separate from ranking. A
startup with a high semantic similarity score must still be marked
ineligible if it fails a mandatory requirement (e.g. missing DPIIT
recognition). A high score never overrides a failed hard requirement.
"""

import re

import pandas as pd


def parse_requirements(requirement_text: str) -> dict:
    """Parse a challenge's mandatory_requirements text into a structured
    dict of individual checks.

    Returns a dict with keys:
    - dpiit_required: bool
    - certification_required: str or None (e.g. "ISO 27001")
    - min_experience_years: int or None

    This uses simple keyword/pattern matching rather than a full NLP
    parser - our requirement text follows a small number of known
    patterns, so this is intentionally simple and easy to extend.
    """
    text = str(requirement_text).lower()

    requirements = {
        "dpiit_required": False,
        "certification_required": None,
        "min_experience_years": None,
    }

    if "dpiit" in text:
        requirements["dpiit_required"] = True

    cert_match = re.search(r"(iso\s?\d+|cmmi level \d+|gdpr compliant)", text)
    if cert_match:
        requirements["certification_required"] = cert_match.group(1).upper()

    exp_match = re.search(r"minimum\s+(\d+)\s+years?", text)
    if exp_match:
        requirements["min_experience_years"] = int(exp_match.group(1))

    return requirements


def check_eligibility(startup_row: pd.Series, requirements: dict) -> tuple:
    """Check one startup against a parsed requirements dict.

    Returns (is_eligible, reasons) where reasons is a list of strings
    explaining any failed requirement (empty list if fully eligible).
    """
    reasons = []

    if requirements["dpiit_required"] and not bool(startup_row["dpiit_recognized"]):
        reasons.append("Missing required DPIIT recognition")

    if requirements["certification_required"]:
        required_cert = requirements["certification_required"]
        startup_certs = str(startup_row["certifications"]).upper()
        if required_cert not in startup_certs:
            reasons.append(f"Missing required certification: {required_cert}")

    if requirements["min_experience_years"] is not None:
        if startup_row["experience"] < requirements["min_experience_years"]:
            reasons.append(
                f"Insufficient experience: has {startup_row['experience']} years, "
                f"requires {requirements['min_experience_years']}+"
            )

    is_eligible = len(reasons) == 0
    return is_eligible, reasons


def filter_eligible_startups(challenge_row: pd.Series, startups: pd.DataFrame) -> pd.DataFrame:
    """Apply eligibility checks to every startup row given.

    Adds two columns: 'eligible' (bool) and 'ineligibility_reasons' (str).
    Returns ALL rows (not dropped) so callers can see both eligible and
    ineligible candidates - useful for transparency and debugging. The
    caller decides later whether to drop ineligible rows before ranking.
    """
    requirements = parse_requirements(challenge_row["mandatory_requirements"])

    eligible_flags = []
    reasons_list = []

    for _, startup_row in startups.iterrows():
        is_eligible, reasons = check_eligibility(startup_row, requirements)
        eligible_flags.append(is_eligible)
        reasons_list.append("; ".join(reasons) if reasons else "")

    result = startups.copy()
    result["eligible"] = eligible_flags
    result["ineligibility_reasons"] = reasons_list
    return result


if __name__ == "__main__":
    from src.similarity import get_top_candidates, load_ready_data

    startups, challenges = load_ready_data()

    # Challenge 102 requires ISO 27001 - a good test case, since MedAssist
    # Systems (healthcare, AI diagnostics) should score high semantically
    # but has no certifications in our dataset.
    test_challenge_id = 102
    challenge_row = challenges[challenges["challenge_id"] == test_challenge_id].iloc[0]

    print(f"Challenge: {challenge_row['title']}")
    print(f"Mandatory requirements (raw): {challenge_row['mandatory_requirements']}")

    requirements = parse_requirements(challenge_row["mandatory_requirements"])
    print(f"Parsed requirements: {requirements}\n")

    top_candidates = get_top_candidates(challenge_id=test_challenge_id, top_k=20)

    # top_candidates only has startup_id/name/sector/semantic_score - merge
    # back with full startup fields needed for eligibility checks.
    candidates_full = top_candidates.merge(
        startups[["startup_id", "dpiit_recognized", "certifications", "experience"]],
        on="startup_id",
    )

    checked = filter_eligible_startups(challenge_row, candidates_full)

    print("Eligibility check on top 20 semantic candidates:")
    print(
        checked[["startup_id", "name", "semantic_score", "eligible", "ineligibility_reasons"]]
        .to_string(index=False)
    )

    eligible_only = checked[checked["eligible"]]
    print(f"\n{len(eligible_only)} of {len(checked)} candidates are eligible.")