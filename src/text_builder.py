"""
src/text_builder.py

Converts a startup row or a challenge row (from our cleaned CSVs) into a
single structured block of text. This text is what actually gets passed
into the embedding model (src/embeddings.py) — the model needs one
coherent piece of text per item, not a spreadsheet row.

Why labeled fields (e.g. "Sector: ..."):
Combining raw values without labels loses structure. Explicitly labeling
each field ("Technologies: ...", "Experience: ...") helps the embedding
model understand what each piece of information represents, which tends
to produce more reliable semantic matches.
"""

import pandas as pd


def create_startup_text(row: pd.Series) -> str:
    """Build a structured text block describing one startup.

    Combines: name, description, sector, technologies, capabilities,
    experience, previous projects, and location.
    """
    return (
        f"Startup: {row['name']}\n"
        f"Description: {row['description']}\n"
        f"Sector: {row['sector']}\n"
        f"Technologies: {row['technologies']}\n"
        f"Capabilities: {row['capabilities']}\n"
        f"Experience: {row['experience']} years\n"
        f"Previous Projects: {row['previous_projects']}\n"
        f"Location: {row['location']}"
    ).strip()


def create_challenge_text(row: pd.Series) -> str:
    """Build a structured text block describing one government challenge.

    Combines: title, problem, sector, required technologies,
    desired outcome, and location.
    """
    return (
        f"Challenge: {row['title']}\n"
        f"Problem: {row['problem']}\n"
        f"Sector: {row['sector']}\n"
        f"Required Technologies: {row['technologies']}\n"
        f"Desired Outcome: {row['desired_outcome']}\n"
        f"Location: {row['location']}"
    ).strip()


def add_embedding_text_column(df: pd.DataFrame, text_fn) -> pd.DataFrame:
    """Apply a text-building function to every row and store it in a new
    'embedding_text' column. Works for either startups or challenges,
    depending on which text_fn is passed in.
    """
    df = df.copy()
    df["embedding_text"] = df.apply(text_fn, axis=1)
    return df


if __name__ == "__main__":
    startups = pd.read_csv("data/startups_clean.csv")
    challenges = pd.read_csv("data/challenges_clean.csv")

    startups = add_embedding_text_column(startups, create_startup_text)
    challenges = add_embedding_text_column(challenges, create_challenge_text)

    print("=" * 60)
    print("SAMPLE STARTUP TEXT (row 0)")
    print("=" * 60)
    print(startups.loc[0, "embedding_text"])

    print("\n" + "=" * 60)
    print("SAMPLE CHALLENGE TEXT (row 0)")
    print("=" * 60)
    print(challenges.loc[0, "embedding_text"])

    startups.to_csv("data/startups_ready.csv", index=False)
    challenges.to_csv("data/challenges_ready.csv", index=False)

    print("\nSaved data/startups_ready.csv and data/challenges_ready.csv")
    print("(each now has an 'embedding_text' column)")