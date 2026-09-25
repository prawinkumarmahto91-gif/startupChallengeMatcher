"""
validate_data.py

Quick sanity checks on the generated datasets.
Run this once after generate_dataset.py, then you can delete it.
"""

import pandas as pd

startups = pd.read_csv("data/startups.csv")
challenges = pd.read_csv("data/challenges.csv")

print("=" * 50)
print("STARTUPS")
print("=" * 50)
print("Shape:", startups.shape)
print("\nColumns:", list(startups.columns))
print("\nMissing values per column:\n", startups.isnull().sum())
print("\nDuplicate startup_id count:", startups["startup_id"].duplicated().sum())
print("\nSample rows:\n", startups.head(3))

print("\n" + "=" * 50)
print("CHALLENGES")
print("=" * 50)
print("Shape:", challenges.shape)
print("\nColumns:", list(challenges.columns))
print("\nMissing values per column:\n", challenges.isnull().sum())
print("\nDuplicate challenge_id count:", challenges["challenge_id"].duplicated().sum())
print("\nSample rows:\n", challenges.head(3))