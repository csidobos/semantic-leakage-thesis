
"""Validate cleaned experiment data without modifying files."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "results" / "clean"

KEY = [
    "model",
    "language",
    "response_mode",
    "repetition",
    "request_id",
]

EXPECTED = 483_840
EXPECTED_PER_GROUP = 20_160

files = sorted(CLEAN.glob("*/*.csv"))

print(f"Clean files found: {len(files)}")

if len(files) != 24:
    raise ValueError("Expected 24 cleaned CSV files.")

df = pd.concat(
    [pd.read_csv(p, dtype={"request_id": str},
                 low_memory=False) for p in files],
    ignore_index=True,
)

print(f"Total rows: {len(df):,}")

if len(df) != EXPECTED:
    raise ValueError("Incorrect total row count.")

duplicates = df.duplicated(KEY).sum()
print(f"Duplicate experiment keys: {duplicates}")

if duplicates:
    raise ValueError("Duplicate experiment keys found.")

counts = df.groupby(
    ["model", "language", "response_mode"]
).size()

if len(counts) != 24 or not counts.eq(EXPECTED_PER_GROUP).all():
    raise ValueError("Incomplete model/language/mode groups.")

print("All 24 groups contain 20,160 unique responses.")

# Validate MC answer-to-category mapping.
mc = df[df["response_mode"].isin(
    ["mc", "mc_unknown"]
)].copy()

allowed_classes = {
    "Sztereotíp",
    "Anti",
    "Semleges",
    "Nem állapítható meg",
    "Cannot be determined",
}

issues = []

for idx, row in mc.iterrows():
    letter = str(row["response_normalized"]).strip()

    allowed_letters = (
        "ABC" if row["response_mode"] == "mc"
        else "ABCD"
    )

    if letter not in allowed_letters:
        issues.append((idx, "invalid_letter"))
        continue

    expected_class = row.get(f"option_{letter}_class")
    actual_class = row["answer_class"]

    if pd.isna(expected_class) or pd.isna(actual_class):
        issues.append((idx, "missing_class"))
    elif str(expected_class) != str(actual_class):
        issues.append((idx, "class_mismatch"))

print(f"\nMC rows checked: {len(mc):,}")
print(f"MC mapping issues: {len(issues):,}")

if issues:
    print("First 10 issues:")
    for idx, reason in issues[:10]:
        row = mc.loc[idx]
        print(
            row["model"],
            row["language"],
            row["response_mode"],
            row["request_id"],
            reason,
        )

print("\nFORMAT QUALITY (SELECTED RESPONSES)")

quality = (
    df.groupby(["model", "language", "response_mode"])
    .agg(
        responses=("request_id", "size"),
        recovered=("recovered_from_failure", "sum"),
        over_10_words=("over_10_words", "sum"),
        over_20_words=("over_20_words", "sum"),
        multiline=("multiline", "sum"),
    )
    .reset_index()
)

print(quality.to_string(index=False))

if issues:
    raise ValueError("MC category mapping validation failed.")

print("\nVALIDATION PASSED")
