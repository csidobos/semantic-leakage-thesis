"""Create thesis bootstrap BCa confidence interval figures.

Dependencies: pandas, altair, vl-convert-python
Run from repository root: python thesis/21_bootstrap_figures.py
"""
from pathlib import Path
import argparse

import altair as alt
import pandas as pd

MODEL_ORDER = ["OpenAI", "Mistral", "Qwen", "Grok"]
LANG_ORDER = ["HU", "EN"]
LANG_COLORS = {"HU": "#477050", "EN": "#3F6FA6"}
CONDITION_ORDER = ["Target", "Comparison", "Implicit", "Nonsocial", "Pseudo", "Nonconnected"]
CONDITION_LABELS = {
    "target": "Target", "comparison": "Comparison", "implicit": "Implicit",
    "nonsocial": "Nonsocial", "pseudo": "Pseudo", "nonconnected": "Nonconnected",
}
MODEL_LABELS = {"openai": "OpenAI", "mistral": "Mistral", "qwen": "Qwen", "grok": "Grok"}


def read_contrasts(path: Path, mode: str, metric: str) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = {"analysis", "metric", "mode", "stratum", "condition", "reference", "estimate_pp", "ci_lower", "ci_upper", "bca_defined"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    data = data.loc[data["mode"].eq(mode) & data["metric"].eq(metric) & data["reference"].eq("neutral")].copy()
    for field in ["estimate_pp", "ci_lower", "ci_upper"]:
        data[field] = pd.to_numeric(data[field], errors="raise")
    data = data.loc[data["bca_defined"].astype(str).str.lower().eq("true")]
    data["condition_label"] = data["condition"].map(CONDITION_LABELS)
    if data["condition_label"].isna().any():
        raise ValueError("Unknown condition labels")
    return data


def common_style(chart: alt.Chart) -> alt.Chart:
    return (chart.configure_view(stroke=None)
            .configure_axis(labelFont="Arial", titleFont="Arial", labelFontSize=12,
                            titleFontSize=12, gridColor="#E6E8EB", domain=False,
                            tickColor="#B8BDC4")
            .configure_header(labelFont="Arial", labelFontSize=14, labelFontWeight="bold")
            .configure_legend(labelFont="Arial", titleFont="Arial", labelFontSize=11))


def overall_figure(data: pd.DataFrame, *, metric_label: str) -> alt.Chart:
    frame = data.loc[data["analysis"].eq("overall") & data["stratum"].eq("all")].copy()
    if len(frame) != 6:
        raise ValueError(f"Expected 6 overall conditions, found {len(frame)}")
    base = alt.Chart(frame).encode(
        y=alt.Y("condition_label:N", sort=CONDITION_ORDER, title=None,
                axis=alt.Axis(labelFontSize=12)),
        x=alt.X("estimate_pp:Q", title=metric_label,
                axis=alt.Axis(labelFontSize=11)),
        tooltip=[alt.Tooltip("condition_label:N", title="Kontextus"),
                 alt.Tooltip("estimate_pp:Q", title="Különbség (pp)", format="+.2f"),
                 alt.Tooltip("ci_lower:Q", title="95% BCa alsó", format=".2f"),
                 alt.Tooltip("ci_upper:Q", title="95% BCa felső", format=".2f")],
    )
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color="#80868B", strokeDash=[4, 4]).encode(x="zero:Q")
    intervals = base.mark_rule(strokeWidth=2.5, color="#68727B").encode(x="ci_lower:Q", x2="ci_upper:Q")
    dots = base.mark_point(filled=True, size=100, color="#477050")
    return common_style((zero + intervals + dots).properties(width=610, height=255))


def model_language_figure(data: pd.DataFrame) -> alt.Chart:
    frame = data.loc[data["analysis"].eq("model_language") & data["condition"].eq("target")].copy()
    if len(frame) != 8:
        raise ValueError(f"Expected 8 model-language Target contrasts, found {len(frame)}")
    parts = frame["stratum"].str.split("|", regex=False, expand=True)
    frame["model"] = parts[0].map(MODEL_LABELS)
    frame["language"] = parts[1]
    if frame["model"].isna().any() or not frame["language"].isin(LANG_ORDER).all():
        raise ValueError("Unexpected model-language stratum")
    base = alt.Chart(frame).encode(
        y=alt.Y("model:N", sort=MODEL_ORDER, title=None),
        x=alt.X("estimate_pp:Q", title="Target − Neutral (százalékpont)"),
        color=alt.Color("language:N", scale=alt.Scale(domain=LANG_ORDER,
                        range=[LANG_COLORS[k] for k in LANG_ORDER]), legend=None),
        tooltip=[alt.Tooltip("model:N", title="Modell"),
                 alt.Tooltip("language:N", title="Nyelv"),
                 alt.Tooltip("estimate_pp:Q", title="Különbség (pp)", format="+.2f"),
                 alt.Tooltip("ci_lower:Q", title="95% BCa alsó", format=".2f"),
                 alt.Tooltip("ci_upper:Q", title="95% BCa felső", format=".2f")],
    )
    intervals = base.mark_rule(strokeWidth=2.5).encode(x="ci_lower:Q", x2="ci_upper:Q")
    dots = base.mark_point(filled=True, size=95)
    layered = (intervals + dots).properties(width=380, height=235)
    return common_style(layered.facet(
        column=alt.Column("language:N", sort=LANG_ORDER, title=None,
                          header=alt.Header(labelExpr="datum.value === 'HU' ? 'Magyar' : 'Angol'"))
    ))


def save_figure(chart: alt.Chart, output_dir: Path, basename: str) -> None:
    for extension in ("png",):
        output = output_dir / f"{basename}.{extension}"
        chart.save(str(output))
        print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/analysis/bootstrap_32_items"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mc = read_contrasts(args.input_dir / "02_neutral_contrasts_bca.csv", "mc", "stereotype")
    naa = read_contrasts(args.input_dir / "05_naa_neutral_contrasts_bca.csv", "mc_unknown", "naa")

    save_figure(overall_figure(mc, metric_label="Különbség a Neutral feltételhez képest (százalékpont)"),
                args.output_dir, "07_bootstrap_mc_context_vs_neutral")
    save_figure(model_language_figure(mc), args.output_dir, "08_bootstrap_target_neutral_model_language")
    save_figure(overall_figure(naa, metric_label="Különbség a Neutral feltételhez képest (százalékpont)"),
                args.output_dir, "09_bootstrap_naa_context_vs_neutral")


if __name__ == "__main__":
    main()
