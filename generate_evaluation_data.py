"""
generate_evaluation_data.py

Creates data/evaluation.csv - a ground-truth "gold standard" of which
startups are considered genuinely relevant for each challenge.

IMPORTANT LIMITATION (worth understanding, not hiding):
We don't have real domain-expert labels yet (those would come from
actual SIH judges or government reviewers). As a reasonable stand-in
for the MVP, we define "relevant" using an objective, computable rule:
a startup is relevant to a challenge if it operates in the SAME SECTOR.

This is not a perfect measure of real-world quality, but it gives us a
consistent, honest way to test whether our ranking pipeline behaves
sensibly. Once real expert feedback or pilot outcomes exist (spec
Section 18, Future Improvements), this file should be replaced with
genuine labels - none of the evaluation metric code needs to change.
"""

import pandas as pd

startups = pd.read_csv("data/startups_clean.csv")
challenges = pd.read_csv("data/challenges_clean.csv")

rows = []
for _, challenge in challenges.iterrows():
    same_sector = startups[
        startups["sector"].str.strip().str.lower() == str(challenge["sector"]).strip().lower()
    ]
    for _, startup in same_sector.iterrows():
        rows.append({
            "challenge_id": challenge["challenge_id"],
            "startup_id": startup["startup_id"],
            "relevant": 1,
        })

evaluation_df = pd.DataFrame(rows)
evaluation_df.to_csv("data/evaluation.csv", index=False)

print(f"Created data/evaluation.csv with {len(evaluation_df)} relevant (challenge, startup) pairs")
print(f"Covering {evaluation_df['challenge_id'].nunique()} of {len(challenges)} challenges\n")
print("Relevant startups per challenge:")
print(evaluation_df.groupby("challenge_id").size())