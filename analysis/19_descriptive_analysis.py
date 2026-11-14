"""Reproducible descriptive analysis for MC and MC+NAA responses.

Run from repository root:
    python analysis/19_descriptive_analysis.py

Inputs: results/clean/mc/*.csv and results/clean/mc_unknown/*.csv
Outputs: results/analysis/descriptive/*.csv
No statistical inference is performed.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "results" / "clean"
OUTPUT = ROOT / "results" / "analysis" / "descriptive"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODELS = {"grok", "mistral", "openai", "qwen"}
LANGUAGES = {"HU", "EN"}
MODES = {"mc", "mc_unknown"}
CONDITIONS = {"target", "neutral", "comparison", "pseudo", "nonconnected", "nonsocial", "implicit"}
CLASSES = {"Sztereotíp", "Anti", "Semleges", "Unknown"}
BASE = ["model", "language", "response_mode"]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load_data():
    paths = sorted([* (INPUT / "mc").glob("*.csv"), * (INPUT / "mc_unknown").glob("*.csv")])
    check(len(paths) == 16, f"Expected 16 clean CSVs; found {len(paths)}")
    frames = []
    required = BASE + ["item_id", "group", "domain", "dimension", "condition", "distance", "paraphrase", "repetition", "answer_class"]
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        check(not (set(required) - set(frame.columns)), f"Missing columns in {path.name}: {set(required) - set(frame.columns)}")
        check(len(frame) == 20160, f"Unexpected row count in {path.name}: {len(frame)}")
        check(frame[required].notna().all().all(), f"Missing required values in {path.name}")
        check(frame["model"].nunique() == frame["language"].nunique() == frame["response_mode"].nunique() == 1, f"Mixed model/language/mode in {path.name}")
        mode = frame["response_mode"].iat[0]
        check(mode == path.parent.name, f"Mode/folder mismatch: {path.name}")
        frames.append(frame[required].copy())
    df = pd.concat(frames, ignore_index=True)
    combos = df[BASE].drop_duplicates()
    check(len(combos) == 16, "Expected 16 unique model-language-mode combinations")
    check(set(df.model) == MODELS and set(df.language) == LANGUAGES and set(df.response_mode) == MODES, "Unexpected models/languages/modes")
    check(set(df.condition) == CONDITIONS, "Unexpected context conditions")
    check(set(df.answer_class).issubset(CLASSES), "Unexpected answer class")
    check(not ((df.response_mode == "mc") & (df.answer_class == "Unknown")).any(), "NAA answer found in MC")
    check(set(df.distance) == {0, 1, 2}, "Unexpected distances")
    check(set(df.paraphrase) == {"p0", "p1", "p2"}, "Unexpected paraphrases")
    check(set(df.repetition) == set(range(1, 11)), "Unexpected repetitions")
    check(df.item_id.nunique() == 32, "Expected 32 item IDs")
    check(df.groupby(["language", "item_id"])["group"].nunique().eq(1).all(), "An item maps to multiple groups within a language")
    check(df.groupby(["language", "item_id"])["dimension"].nunique().eq(1).all(), "An item maps to multiple dimensions within a language")
    check(df.groupby(["language", "item_id"])["domain"].nunique().eq(1).all(), "An item maps to multiple domains within a language")
    key = BASE + ["item_id", "condition", "distance", "paraphrase", "repetition"]
    check(not df.duplicated(key).any(), "Duplicate observation keys")
    cell = BASE + ["item_id", "condition", "distance", "paraphrase"]
    sizes = df.groupby(cell, dropna=False).size()
    check(len(sizes) == 4 * 2 * 2 * 32 * 7 * 3 * 3, "Missing experimental cells")
    check(sizes.eq(10).all(), "Expected exactly 10 repetitions per experimental cell")
    check(df.groupby(["model", "language", "response_mode", "condition"]).size().eq(2880).all(), "Unequal context sizes")
    df["is_stereotype"] = df.answer_class.eq("Sztereotíp").astype(int)
    df["is_naa"] = df.answer_class.eq("Unknown").astype(int)
    df["is_content"] = df.answer_class.ne("Unknown").astype(int)
    return df


def summarize(df, dimensions):
    counts = df.groupby(dimensions, dropna=False).agg(
        n=("answer_class", "size"),
        stereotype_n=("is_stereotype", "sum"),
        naa_n=("is_naa", "sum"),
        content_n=("is_content", "sum"),
    ).reset_index()
    counts["nonstereotype_n"] = counts.n - counts.stereotype_n
    counts["stereotype_pct"] = 100 * counts.stereotype_n / counts.n
    counts["naa_pct"] = 100 * counts.naa_n / counts.n
    counts["stereotype_content_pct"] = 100 * counts.stereotype_n / counts.content_n.replace(0, float("nan"))
    # NAA is not an available choice in MC mode.
    counts.loc[counts.response_mode.eq("mc"), ["naa_pct", "stereotype_content_pct"]] = float("nan")
    return counts


def contrast(summary, dimensions, reference="neutral", include_reference=False):
    check("condition" in dimensions, "A condition column is required for contrasts")
    pairing = [c for c in dimensions if c != "condition"]
    measures = ["n", "stereotype_n", "naa_n", "content_n", "stereotype_pct", "naa_pct", "stereotype_content_pct"]
    ref = summary.loc[summary.condition.eq(reference), pairing + measures].rename(columns={m: f"{reference}_{m}" for m in measures})
    check(not ref.duplicated(pairing).any(), "Ambiguous reference matches")
    source = summary if include_reference else summary.loc[summary.condition.ne(reference)]
    merged = source.merge(ref, on=pairing, how="left", validate="many_to_one", indicator=True)
    check(merged._merge.eq("both").all(), f"Missing {reference} matches")
    merged = merged.drop(columns="_merge")
    for m in ["stereotype_pct", "naa_pct", "stereotype_content_pct"]:
        merged[f"delta_{m}_pp_vs_{reference}"] = merged[m] - merged[f"{reference}_{m}"]
    return merged


def target_neutral(summary, dimensions):
    check("condition" in dimensions, "Missing condition")
    paired = contrast(summary, dimensions, reference="neutral")
    paired = paired.loc[paired.condition.eq("target")].drop(columns="condition")
    rename = {c: "target_" + c for c in ["n", "stereotype_n", "naa_n", "content_n", "stereotype_pct", "naa_pct", "stereotype_content_pct"]}
    paired = paired.rename(columns=rename)
    return paired


def save(df, filename):
    path = OUTPUT / filename
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.4f")
    print(f"{filename}: {len(df):,} rows")


def main():
    df = load_data()
    print(f"Validated {len(df):,} individual responses")

    # Core descriptive outputs.
    context = summarize(df, BASE + ["condition"])
    save(context, "01_context_rates.csv")
    save(target_neutral(context, BASE + ["condition"]), "02_target_neutral.csv")

    group = summarize(df, BASE + ["group", "condition"])
    save(target_neutral(group, BASE + ["group", "condition"]), "03_group_delta.csv")

    item_keys = BASE + ["domain", "dimension", "group", "item_id", "condition"]
    item = summarize(df, item_keys)
    save(target_neutral(item, item_keys), "04_item_delta.csv")

    save(contrast(context, BASE + ["condition"], reference="neutral"), "05_context_delta_vs_neutral.csv")
    save(contrast(context, BASE + ["condition"], reference="target"), "05b_context_delta_vs_target.csv")

    distance_keys = BASE + ["distance", "condition"]
    distance = summarize(df, distance_keys)
    save(target_neutral(distance, distance_keys), "06_distance_delta.csv")

    paraphrase_keys = BASE + ["paraphrase", "condition"]
    paraphrase = summarize(df, paraphrase_keys)
    save(target_neutral(paraphrase, paraphrase_keys), "07_paraphrase_delta.csv")

    # Item-level rates and comparisons for all seven conditions.
    item_neutral = contrast(item, item_keys, reference="neutral", include_reference=True)
    item_target = contrast(item, item_keys, reference="target", include_reference=True)
    target_cols = [* [c for c in item_keys if c != "condition"], "condition", "target_stereotype_pct", "target_naa_pct", "delta_stereotype_pct_pp_vs_target", "delta_naa_pct_pp_vs_target"]
    item_all = item_neutral.merge(item_target[target_cols], on=item_keys, validate="one_to_one")
    save(item_all, "08_item_context_rates_and_deltas.csv")

    # Detailed views retain distance/paraphrase rather than hiding them.
    item_distance_keys = BASE + ["domain", "dimension", "group", "item_id", "distance", "condition"]
    item_distance = summarize(df, item_distance_keys)
    save(target_neutral(item_distance, item_distance_keys), "09_item_distance_target_neutral.csv")

    item_para_keys = BASE + ["domain", "dimension", "group", "item_id", "paraphrase", "condition"]
    item_para = summarize(df, item_para_keys)
    save(target_neutral(item_para, item_para_keys), "10_item_paraphrase_target_neutral.csv")

    # A fully disaggregated summary; only the 10 repetitions are combined.
    detail_keys = BASE + ["domain", "dimension", "group", "item_id", "condition", "distance", "paraphrase"]
    detailed = summarize(df, detail_keys)
    check(detailed.n.eq(10).all(), "Detailed rows must each contain 10 responses")
    save(detailed, "11_full_cells_counts_and_rates.csv")

    print("DESCRIPTIVE ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
