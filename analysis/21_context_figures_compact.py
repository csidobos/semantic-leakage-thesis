"""Plot MC stereotype response rates by context, model, and language.

Run from repository root: python analysis/21_context_figures_compact.py
Dependencies: pandas, altair, vl-convert-python
"""
from pathlib import Path
import argparse
import pandas as pd
import altair as alt

MODEL_ORDER = ["OpenAI", "Mistral", "Qwen", "Grok"]
MODEL_NAMES = {"openai": "OpenAI", "mistral": "Mistral", "qwen": "Qwen", "grok": "Grok"}
CONDITION_ORDER = ["Target", "Comparison", "Neutral", "Pseudo", "Nonconnected", "Nonsocial", "Implicit"]
LANGUAGE_ORDER = ["HU", "EN"]
LANGUAGE_COLORS = ["#477050", "#3F6FA6"]


def make_figure(data: pd.DataFrame) -> alt.Chart:
    required = {"model", "language", "response_mode", "condition", "stereotype_pct"}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frame = data.loc[data["response_mode"].eq("mc")].copy()
    frame["model"] = frame["model"].map(MODEL_NAMES)
    frame["condition"] = frame["condition"].str.lower().map(
        {condition.lower(): condition for condition in CONDITION_ORDER}
    )
    frame = frame.loc[frame["model"].isin(MODEL_ORDER) & frame["language"].isin(LANGUAGE_ORDER)]
    frame["stereotype_pct"] = pd.to_numeric(frame["stereotype_pct"], errors="raise")
    if frame["condition"].isna().any():
        raise ValueError("Unrecognized context condition")
    expected = len(MODEL_ORDER) * len(LANGUAGE_ORDER) * len(CONDITION_ORDER)
    if len(frame) != expected or frame.duplicated(["model", "language", "condition"]).any():
        raise ValueError(f"Expected {expected} unique rows, found {len(frame)}")

    frame["language_label"] = frame["language"].map({"HU": "Magyar", "EN": "Angol"})

    def panel(model: str) -> alt.Chart:
        subset = frame.loc[frame["model"].eq(model)]
        return (
            alt.Chart(subset)
            .mark_bar()
            .encode(
                x=alt.X(
                    "condition:N", sort=CONDITION_ORDER, title=None,
                    axis=alt.Axis(labelAngle=-35, labelFontSize=20, labelLimit=200, labelPadding=8,
                                  labelOverlap=False, labels=True),
                ),
                xOffset=alt.XOffset("language:N", sort=LANGUAGE_ORDER),
                y=alt.Y(
                    "stereotype_pct:Q", title="Sztereotipikus válaszok (%)",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(values=[25, 50, 75, 100], labelFontSize=18, titleFontSize=20, titlePadding=10),
                ),
                color=alt.Color(
                    "language:N", sort=LANGUAGE_ORDER,
                    scale=alt.Scale(domain=LANGUAGE_ORDER, range=LANGUAGE_COLORS),
                    legend=alt.Legend(title="Nyelv", labelFontSize=19, titleFontSize=20, symbolSize=220, labelExpr="datum.label == 'HU' ? 'Magyar' : 'Angol'"),
                ),
                tooltip=[
                    alt.Tooltip("model:N", title="Modell"),
                    alt.Tooltip("condition:N", title="Kontextus"),
                    alt.Tooltip("language_label:N", title="Nyelv"),
                    alt.Tooltip("stereotype_pct:Q", title="Sztereotipikus válaszok (%)", format=".1f"),
                ],
            )
            .properties(width=470, height=240, title=model)
        )

    figure = alt.vconcat(
        alt.hconcat(panel("OpenAI"), panel("Mistral"), spacing=25),
        alt.hconcat(panel("Qwen"), panel("Grok"), spacing=25),
        spacing=35,
    )
    return (
        figure.configure_view(stroke=None)
        .configure_axis(labelFont="Arial", titleFont="Arial", labelFontSize=18, titleFontSize=20,
                        gridColor="#E6E8EB", domain=False)
        .configure_title(font="Arial", fontSize=24, anchor="middle")
        .configure_legend(labelFont="Arial", titleFont="Arial", labelFontSize=19, titleFontSize=20)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("results/analysis/descriptive/01_context_rates.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/analysis/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart = make_figure(pd.read_csv(args.input, encoding="utf-8-sig"))
    for extension in ("svg", "png", "html"):
        path = args.output_dir / f"02_mc_context_rates.{extension}"
        chart.save(str(path))
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
