
"""Clean and audit the four-model semantic leakage experiment.

Run from the project root:
    python analysis/04_clean_data.py

Inputs:
    results/raw/*.csv
    results/grok/raw/*.csv

Outputs:
    results/clean/<mode>/<model>_<language>_<mode>.csv
    results/audit/cleaning_summary.csv
    results/audit/response_quality_raw_attempts.csv
    results/audit/cleaning_excluded_rows.csv

Raw files are never modified. No API calls are made.
"""

import sys
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_grok_experiment_fixed import (
    normalize_answer,
    is_valid_answer,
)

SOURCES = [
    PROJECT_ROOT / "results/raw",
    PROJECT_ROOT / "results/grok/raw",
]

OUTPUT = PROJECT_ROOT / "results/clean"
AUDIT = PROJECT_ROOT / "results/audit"

EXPECTED_TOTAL = 483_840
EXPECTED_PER_COMBINATION = 20_160

MODELS = ("grok", "mistral", "openai", "qwen")
LANGUAGES = ("EN", "HU")
MODES = ("free_text", "mc", "mc_unknown")

KEY = [
    "model",
    "language",
    "response_mode",
    "repetition",
    "request_id",
]


# --------------------------------------------------
# 1. IDENTIFY FILE METADATA
# --------------------------------------------------

def identify_mode(path):
    name = path.stem.lower()

    for mode in ("mc_unknown", "free_text", "mc"):
        if f"_{mode}_rep" in name:
            return mode

    raise ValueError(
        f"Cannot identify response mode: {path}"
    )


def identify_model(path):
    name = path.stem.lower()

    for model in MODELS:
        if name.startswith(f"grok_") and model == "grok":
            return model
        if f"_{model}_" in name:
            return model

    raise ValueError(
        f"Cannot identify model: {path}"
    )


# --------------------------------------------------
# 2. NORMALIZATION
# --------------------------------------------------

def normalize_mc(raw, mode):
    """Use exactly the Grok runner's MC validity rules."""

    if pd.isna(raw):
        return None

    raw = str(raw)

    if not is_valid_answer(raw, mode):
        return None

    normalized = normalize_answer(raw, mode)

    if not isinstance(normalized, str):
        return None

    normalized = normalized.strip().upper()

    allowed = "ABC" if mode == "mc" else "ABCD"

    if re.fullmatch(f"[{allowed}]", normalized):
        return normalized

    return None


def normalize_free_text(raw):
    """Preserve substantive free-text responses unchanged."""

    if pd.isna(raw):
        return None

    text = str(raw)

    if not text.strip():
        return None

    return text


# --------------------------------------------------
# 3. FORMAT QUALITY
# --------------------------------------------------

def classify_format(raw, mode):

    if pd.isna(raw) or not str(raw).strip():
        return "empty"

    text = str(raw).strip()

    if mode == "free_text":

        words = len(text.split())

        if words > 20:
            return "over_20_words"

        if words > 10:
            return "over_10_words"

        if "\n" in text:
            return "multiline"

        return "short"

    allowed = "ABC" if mode == "mc" else "ABCD"

    if re.fullmatch(f"[{allowed}]", text):
        return "exact"

    if re.fullmatch(
        rf"\*\*([{allowed}])\*\*",
        text,
        flags=re.IGNORECASE,
    ):
        return "bold_letter"

    if normalize_mc(text, mode) is not None:
        return "other_normalizable"

    return "invalid_or_ambiguous"


# --------------------------------------------------
# 4. READ ONE FILE
# --------------------------------------------------

def process_file(path):

    df = pd.read_csv(
        path,
        dtype={"request_id": str},
        low_memory=False,
    )

    mode = identify_mode(path)
    model = identify_model(path)

    df["model"] = model
    df["response_mode"] = mode
    df["source_file"] = path.name

    df["success_original"] = (
        df["success"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    df["format_type"] = df["response_raw"].apply(
        lambda raw: classify_format(raw, mode)
    )

    df["response_word_count"] = (
        df["response_raw"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    df["over_10_words"] = (
        df["response_word_count"] > 10
    )

    df["over_20_words"] = (
        df["response_word_count"] > 20
    )

    df["multiline"] = (
        df["response_raw"]
        .fillna("")
        .astype(str)
        .str.contains("\n", regex=False)
    )

    if mode in ("mc", "mc_unknown"):

        df["response_normalized"] = (
            df["response_raw"].apply(
                lambda raw: normalize_mc(raw, mode)
            )
        )

    else:

        df["response_normalized"] = (
            df["response_raw"].apply(
                normalize_free_text
            )
        )

    df["response_usable"] = (
        df["response_normalized"].notna()
    )

    error = (
        df["error_message"]
        .fillna("")
        .astype(str)
        if "error_message" in df.columns
        else pd.Series("", index=df.index)
    )

    # Only recover failed MC responses when the recorded
    # failure was explicitly a response-format validation error.
    df["recovered_from_failure"] = (
        ~df["success_original"]
        & df["response_usable"]
        & error.str.match(
            r"^(?:ValueError:\s*)?Invalid "
        )
        & df["response_mode"].isin(
            ("mc", "mc_unknown")
        )
    )

    df["eligible"] = (
        df["response_usable"]
        & (
            df["success_original"]
            | df["recovered_from_failure"]
        )
    )

    # Strict MC format: exactly one allowed letter.
    if mode in ("mc", "mc_unknown"):
        allowed = "ABC" if mode == "mc" else "ABCD"

        df["format_compliant"] = (
            df["response_raw"]
            .fillna("")
            .astype(str)
            .str.fullmatch(f"[{allowed}]")
        )
    else:
        # Free-text format compliance needs separate
        # substantive coding, not a hard word threshold.
        df["format_compliant"] = pd.NA

    # Map selected MC letter to the randomized option class.
    if mode in ("mc", "mc_unknown"):

        def classify_answer(row):

            letter = row["response_normalized"]

            if not isinstance(letter, str):
                return None

            return row.get(f"option_{letter}_class")

        df["answer_class"] = df.apply(
            classify_answer,
            axis=1,
        )

    else:
        df["answer_class"] = None

    return df


# --------------------------------------------------
# 5. MAIN
# --------------------------------------------------

def main():

    OUTPUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    files = [
        path
        for folder in SOURCES
        for path in sorted(folder.glob("*.csv"))
    ]

    print(f"Reading {len(files)} CSV files...")

    if len(files) != 240:
        raise ValueError(
            f"Expected 240 CSV files, found {len(files)}."
        )

    frames = [
        process_file(path)
        for path in files
    ]

    raw = pd.concat(
        frames,
        ignore_index=True,
    )

    print(f"Raw rows: {len(raw):,}")

    if raw[KEY].isna().any().any():
        raise ValueError(
            "Missing values in experiment keys."
        )

    # ----------------------------------------------
    # RAW ATTEMPT QUALITY
    # ----------------------------------------------

    quality = (
        raw.groupby(
            ["model", "language", "response_mode"],
            dropna=False,
        )
        .agg(
            raw_rows=("request_id", "size"),
            original_success=(
                "success_original", "sum"
            ),
            usable_attempts=(
                "response_usable", "sum"
            ),
            recovered_attempts=(
                "recovered_from_failure", "sum"
            ),
            exact_format=(
                "format_compliant", "sum"
            ),
            over_10_words=(
                "over_10_words", "sum"
            ),
            over_20_words=(
                "over_20_words", "sum"
            ),
            multiline=(
                "multiline", "sum"
            ),
        )
        .reset_index()
    )

    # ----------------------------------------------
    # SELECT ONE RESPONSE PER REQUEST
    # ----------------------------------------------

    eligible = raw.loc[
        raw["eligible"]
    ].copy()

    eligible["priority"] = (
        eligible["success_original"].astype(int)
    )

    eligible["finished_sort"] = (
        pd.to_datetime(
            eligible["finished_at"],
            errors="coerce",
            utc=True,
        )
    )

    # Prefer originally successful responses.
    # Within equal priority, prefer the latest attempt.
    eligible = eligible.sort_values(
        ["priority", "finished_sort"],
        kind="stable",
        na_position="first",
    )

    clean = eligible.drop_duplicates(
        subset=KEY,
        keep="last",
    ).copy()

    attempt_counts = (
        raw.groupby(KEY)
        .size()
        .rename("raw_attempt_rows")
        .reset_index()
    )

    clean = clean.merge(
        attempt_counts,
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    clean["multiple_attempts"] = (
        clean["raw_attempt_rows"] > 1
    )

    # ----------------------------------------------
    # AUDIT EXCLUDED ROWS
    # ----------------------------------------------

    excluded = raw.loc[
        ~raw["eligible"]
    ].copy()

    excluded.to_csv(
        AUDIT / "cleaning_excluded_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ----------------------------------------------
    # FINAL QUALITY SUMMARY
    # ----------------------------------------------

    summary = (
        clean.groupby(
            ["model", "language", "response_mode"],
            dropna=False,
        )
        .agg(
            unique_responses=("request_id", "size"),
            recovered_selected=(
                "recovered_from_failure", "sum"
            ),
            multiple_attempts=(
                "multiple_attempts", "sum"
            ),
            exact_format=(
                "format_compliant", "sum"
            ),
            over_10_words=(
                "over_10_words", "sum"
            ),
            over_20_words=(
                "over_20_words", "sum"
            ),
            multiline=(
                "multiline", "sum"
            ),
        )
        .reset_index()
    )

    quality.to_csv(
        AUDIT / "response_quality_raw_attempts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        AUDIT / "cleaning_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ----------------------------------------------
    # VALIDATE BEFORE SAVING CLEAN FILES
    # ----------------------------------------------

    print("\nCLEANING SUMMARY")
    print("-" * 75)
    print(summary.to_string(index=False))

    print(
        f"\nClean unique responses: {len(clean):,}"
    )

    print(
        "Selected recovered responses:",
        int(
            clean["recovered_from_failure"].sum()
        ),
    )

    expected_groups = (
        len(MODELS)
        * len(LANGUAGES)
        * len(MODES)
    )

    if len(summary) != expected_groups:
        raise ValueError(
            f"Expected {expected_groups} combinations, "
            f"found {len(summary)}."
        )

    incomplete = summary.loc[
        summary["unique_responses"]
        != EXPECTED_PER_COMBINATION
    ]

    if not incomplete.empty:

        print("\nINCOMPLETE COMBINATIONS")
        print(incomplete.to_string(index=False))

    if (
        len(clean) != EXPECTED_TOTAL
        or not incomplete.empty
    ):
        raise ValueError(
            "Completeness check failed. "
            "Clean files were not overwritten."
        )

    # ----------------------------------------------
    # SAVE CLEAN FILES
    # ----------------------------------------------

    for (
        model,
        language,
        mode,
    ), subset in clean.groupby(
        ["model", "language", "response_mode"]
    ):

        destination = (
            OUTPUT
            / mode
            / f"{model}_{language}_{mode}.csv"
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        subset.drop(
            columns=["priority", "finished_sort"],
            errors="ignore",
        ).to_csv(
            destination,
            index=False,
            encoding="utf-8-sig",
        )

    print("\nCOMPLETENESS CHECK PASSED")
    print(
        f"{EXPECTED_TOTAL:,} unique responses."
    )
    print("Raw CSV files were not modified.")
    print(f"Clean files saved to: {OUTPUT}")
    print(f"Audit files saved to: {AUDIT}")


if __name__ == "__main__":
    main()
