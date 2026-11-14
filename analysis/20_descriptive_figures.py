"""Create the MC Target–Neutral stereotype-rate figure for the thesis.

Dependencies: pandas, altair, vl-convert-python
Run from repository root: python analysis/20_descriptive_figures.py
"""
from pathlib import Path
import argparse

import altair as alt
import pandas as pd

MODEL_ORDER = ["OpenAI", "Mistral", "Qwen", "Grok"]
LANG_ORDER = ["HU", "EN"]
LANG_COLORS = {"HU": "#477050", "EN": "#3F6FA6"}
CONDITION_ORDER = ["Neutral", "Target"]


def make_target_neutral_figure(data: pd.DataFrame) -> alt.Chart:
    required = {
        "model", "language", "response_mode", "target_stereotype_pct",
        "neutral_stereotype_pct", "delta_stereotype_pct_pp_vs_neutral",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frame = data.loc[data["response_mode"].eq("mc")].copy()
    frame["model"] = frame["model"].str.lower().map({
        "openai": "OpenAI", "mistral": "Mistral", "qwen": "Qwen", "grok": "Grok",
    })
    frame = frame.loc[frame["model"].notna() & frame["language"].isin(LANG_ORDER)]
    frame["delta_pp"] = pd.to_numeric(frame["delta_stereotype_pct_pp_vs_neutral"])
    frame["delta_label"] = frame["delta_pp"].map(lambda x: f"+{x:.1f} pp" if x >= 0 else f"{x:.1f} pp")

    long = frame.melt(
        id_vars=["model", "language", "delta_pp", "delta_label"],
        value_vars=["neutral_stereotype_pct", "target_stereotype_pct"],
        var_name="source", value_name="rate",
    )
    long["condition"] = long["source"].map({
        "neutral_stereotype_pct": "Neutral", "target_stereotype_pct": "Target",
    })
    long["rate"] = pd.to_numeric(long["rate"])
    long["language_name"] = long["language"].map({"HU": "Magyar", "EN": "Angol"})

    base = alt.Chart(long).encode(
        x=alt.X("model:N", sort=MODEL_ORDER, title=None,
                axis=alt.Axis(labelAngle=0, labelFontSize=12)),
        xOffset=alt.XOffset("condition:N", sort=CONDITION_ORDER, title=None),
        y=alt.Y("rate:Q", title="Sztereotipikus válaszok aránya (%)",
                scale=alt.Scale(domain=[0, 85]), axis=alt.Axis(values=[0, 20, 40, 60, 80])),
        color=alt.Color("language:N", scale=alt.Scale(
            domain=LANG_ORDER, range=[LANG_COLORS[k] for k in LANG_ORDER]
        ), legend=None),
        opacity=alt.Opacity("condition:N", sort=CONDITION_ORDER,
                            scale=alt.Scale(domain=CONDITION_ORDER, range=[0.45, 1.0]),
                            legend=alt.Legend(title="Kontextus", orient="top")),
        tooltip=[
            alt.Tooltip("model:N", title="Modell"),
            alt.Tooltip("language_name:N", title="Nyelv"),
            alt.Tooltip("condition:N", title="Kontextus"),
            alt.Tooltip("rate:Q", title="Sztereotipikus válasz (%)", format=".1f"),
            alt.Tooltip("delta_pp:Q", title="Target–Neutral (pp)", format="+.1f"),
        ],
    )
    bars = base.mark_bar(size=24)
    values = base.mark_text(dy=-8, fontSize=10, opacity=1, color="#30343B").encode(
        text=alt.Text("rate:Q", format=".1f")
    )
    chart = (bars + values).properties(width=370, height=320)
    figure = chart.facet(
        column=alt.Column("language:N", sort=LANG_ORDER, title=None,
                          header=alt.Header(labelExpr="datum.value === 'HU' ? 'Magyar' : 'Angol'",
                                            labelFontSize=15, labelFontWeight="bold"))
    )
    return figure.configure_view(stroke=None).configure_axis(
        labelFont="Arial", titleFont="Arial", titleFontSize=12,
        gridColor="#E6E8EB", domain=False, tickColor="#B8BDC4",
    ).configure_legend(labelFont="Arial", titleFont="Arial", labelFontSize=11)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("results/analysis/descriptive/02_target_neutral.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/analysis/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart = make_target_neutral_figure(pd.read_csv(args.input, encoding="utf-8-sig"))
    for extension in ("svg", "png", "html"):
        output = args.output_dir / f"01_mc_target_neutral_rates.{extension}"
        chart.save(str(output))
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
