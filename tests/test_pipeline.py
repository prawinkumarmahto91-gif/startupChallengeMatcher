"""
tests/test_pipeline.py

Automated tests for the matching pipeline. Run with:
    pytest tests/ -v

These are split into two groups:
1. FAST unit tests - test individual functions with made-up data,
   no model loading, run in milliseconds.
2. SLOW integration tests - test the real end-to-end pipeline,
   load the actual model and data, take a few seconds each.
"""

import pandas as pd
import pytest

from src.features import technology_match, sector_match, budget_score, location_score
from src.eligibility import parse_requirements, check_eligibility
from src.matcher import match_startups


# ---------------------------------------------------------------------------
# FAST unit tests - pure functions, fake data, no model needed
# ---------------------------------------------------------------------------

def test_technology_match_full_overlap():
    """Startup covers 100% of the challenge's required technologies."""
    score = technology_match("AI, GPS, IoT", "AI, GPS, IoT")
    assert score == 1.0


def test_technology_match_no_overlap():
    """Completely different technologies -> zero match."""
    score = technology_match("Blockchain, Fintech ML", "AI, GPS, IoT")
    assert score == 0.0


def test_technology_match_partial_overlap():
    """Startup covers 2 of 3 required technologies -> 0.67."""
    score = technology_match("AI, GPS, Blockchain", "AI, GPS, IoT")
    assert round(score, 2) == 0.67


def test_sector_match_same_sector_case_insensitive():
    assert sector_match("Healthcare", "healthcare") == 1.0


def test_sector_match_different_sector():
    assert sector_match("Healthcare", "Fintech") == 0.0


def test_budget_score_within_budget():
    """Startup cost <= challenge budget -> perfect score."""
    assert budget_score(startup_budget=40, challenge_budget=50) == 1.0


def test_budget_score_over_budget():
    """Startup cost 50% over budget -> score should drop meaningfully."""
    score = budget_score(startup_budget=75, challenge_budget=50)
    assert 0.0 <= score < 1.0


def test_location_score_same_city():
    assert location_score("Pune", "Pune") == 1.0


def test_location_score_different_city():
    assert location_score("Pune", "Mumbai") == 0.4


def test_parse_requirements_detects_dpiit():
    reqs = parse_requirements("DPIIT recognition required")
    assert reqs["dpiit_required"] is True


def test_parse_requirements_detects_certification():
    reqs = parse_requirements("ISO 27001 certification required")
    assert reqs["certification_required"] == "ISO 27001"


def test_check_eligibility_fails_without_dpiit():
    """A startup without DPIIT recognition must fail when DPIIT is required."""
    fake_startup = pd.Series({
        "dpiit_recognized": False,
        "certifications": "ISO 9001",
        "experience": 5,
    })
    requirements = {"dpiit_required": True, "certification_required": None, "min_experience_years": None}

    is_eligible, reasons = check_eligibility(fake_startup, requirements)

    assert is_eligible is False
    assert len(reasons) == 1
    assert "DPIIT" in reasons[0]


def test_check_eligibility_passes_when_all_requirements_met():
    fake_startup = pd.Series({
        "dpiit_recognized": True,
        "certifications": "ISO 27001",
        "experience": 8,
    })
    requirements = {"dpiit_required": True, "certification_required": "ISO 27001", "min_experience_years": 3}

    is_eligible, reasons = check_eligibility(fake_startup, requirements)

    assert is_eligible is True
    assert reasons == []


# ---------------------------------------------------------------------------
# SLOW integration tests - real pipeline, real data, real model
# These take longer since they load BGE-M3 and run full retrieval+ranking.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_match_startups_returns_ranked_recommendations():
    """End-to-end: a known-good challenge should return recommendations,
    sorted by descending final_score, with all required fields present.
    """
    result = match_startups(challenge_id=101)

    assert result["challenge_id"] == 101
    assert len(result["recommendations"]) > 0

    scores = [rec["final_score"] for rec in result["recommendations"]]
    assert scores == sorted(scores, reverse=True), "Recommendations must be sorted highest score first"

    top = result["recommendations"][0]
    required_fields = {
        "startup_id", "startup_name", "rank", "final_score", "semantic_score",
        "technology_match", "sector_match", "experience_score",
        "budget_score", "location_score", "reasons",
    }
    assert required_fields.issubset(top.keys())
    assert len(top["reasons"]) == 6  # one reason per scoring dimension


@pytest.mark.slow
def test_match_startups_invalid_challenge_raises_value_error():
    """An unknown challenge_id must raise a clear error, not crash silently."""
    with pytest.raises(ValueError, match="No challenge found"):
        match_startups(challenge_id=999999)


@pytest.mark.slow
def test_eligibility_actually_excludes_ineligible_startups():
    """Sanity check on real data: challenge 102 requires ISO 27001.
    No recommended startup should be missing that certification.
    """
    result = match_startups(challenge_id=102)

    for rec in result["recommendations"]:
        # If a startup made it into the final list, it must have passed
        # eligibility - meaning it should NOT show a certification failure
        # reason (this indirectly confirms filtering happened correctly).
        assert not any("Missing required certification" in r for r in rec["reasons"] if "certification" in r.lower() and "Missing" in r)