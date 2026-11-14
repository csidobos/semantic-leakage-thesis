"""Plot MC+NAA response composition in Target and Neutral contexts.

Run from repository root: python analysis/25_mc_naa_figures.py
Dependencies: pandas, altair, vl-convert-python
"""
from pathlib import Path
import argparse
import pandas as pd
import altair as alt

MODEL_ORDER = ["OpenAI", "Mistral", "Qwen", "Grok"]
MODEL_NAMES = {"openai": "OpenAI", "mistral": "Mistral", "qwen": "Qwen", "grok": "Grok"}
CONDITION_ORDER = ["Target", "Neutral"]
CATEGORY_ORDER = ["Sztereotipikus", "Nem sztereotipikus", "Nem állapítható meg (NAA)"]
PALETTES = {
    "HU": ["#477050", "#B9D1BD", "#E1E3E6"],
    "EN": ["#3F6FA6", "#B8CEE7", "#E1E3E6"],
}


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "language", "response_mode", "condition", "n", "stereotype_n", "naa_n"}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = data.loc[
        data["response_mode"].eq("mc_unknown")
        & data["condition"].isin(["target", "neutral"])
    ].copy()
    if len(frame) != 16 or frame.duplicated(["model", "language", "condition"]).any():
        raise ValueError("Expected 16 unique model-language-condition rows for MC+NAA")
    frame["model"] = frame["model"].map(MODEL_NAMES)
    if frame["model"].isna().any() or set(frame["language"]) != {"HU", "EN"}:
        raise ValueError("Unexpected model or language")
    for column in ("n", "stereotype_n", "naa_n"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["other_content_n"] = frame["n"] - frame["stereotype_n"] - frame["naa_n"]
    if (frame[["stereotype_n", "naa_n", "other_content_n"]] < 0).any().any():
        raise ValueError("Inconsistent response counts")
    frame["condition"] = frame["condition"].map({"target": "Target", "neutral": "Neutral"})
    long = frame.melt(
        id_vars=["model", "language", "condition", "n"],
        value_vars=["stereotype_n", "other_content_n", "naa_n"],
        var_name="category", value_name="count",
    )
    long["category"] = long["category"].map({
        "stereotype_n": "Sztereotipikus",
        "other_content_n": "Nem sztereotipikus",
        "naa_n": "Nem állapítható meg (NAA)",
    })
    long["percent"] = 100 * long["count"] / long["n"]
    return long


def make_figure(data: pd.DataFrame, language: str) -> alt.Chart:
    subset = data.loc[data["language"].eq(language)]

    def panel(model: str) -> alt.Chart:
        source = subset.loc[subset["model"].eq(model)]
        return alt.Chart(source).mark_bar(size=78).encode(
            x=alt.X("condition:N", sort=CONDITION_ORDER, title=None,
                    axis=alt.Axis(labelAngle=0, labelFontSize=14)),
            y=alt.Y("percent:Q", stack="zero", title="Válaszok aránya (%)",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(values=[0, 25, 50, 75, 100], labelFontSize=12, titleFontSize=13)),
            color=alt.Color("category:N", sort=CATEGORY_ORDER,
                            scale=alt.Scale(domain=CATEGORY_ORDER, range=PALETTES[language]),
                            legend=alt.Legend(title="Válaszkategória", orient="bottom",
                                              labelFontSize=12, titleFontSize=13)),
            order=alt.Order("category:N", sort="ascending"),
            tooltip=[alt.Tooltip("model:N", title="Modell"),
                     alt.Tooltip("condition:N", title="Kontextus"),
                     alt.Tooltip("category:N", title="Válasz"),
                     alt.Tooltip("count:Q", title="Darab"),
                     alt.Tooltip("percent:Q", title="Arány (%)", format=".2f")],
        ).properties(width=300, height=250, title=model)

    top = alt.hconcat(panel("OpenAI"), panel("Mistral"), spacing=55)
    bottom = alt.hconcat(panel("Qwen"), panel("Grok"), spacing=55)
    return (alt.vconcat(top, bottom, spacing=50)
            .resolve_scale(color="shared")
            .properties(title=f"MC+NAA válaszmegoszlások – {'magyar' if language == 'HU' else 'angol'}")
            .configure_view(stroke=None)
            .configure_axis(labelFont="Arial", titleFont="Arial", gridColor="#E6E8EB", domain=False)
            .configure_title(font="Arial", fontSize=18, anchor="middle")
            .configure_legend(labelFont="Arial", titleFont="Arial"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/analysis/descriptive/01_context_rates.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(pd.read_csv(args.input, encoding="utf-8-sig"))
    for language in ("HU", "EN"):
        chart = make_figure(data, language)
        for extension in ("svg", "png", "html"):
            output = args.output_dir / f"06_mc_naa_response_distribution_{language.lower()}.{extension}"
            chart.save(str(output))
            print(f"Saved: {output}")


if __name__ == "__main__":
    main()
