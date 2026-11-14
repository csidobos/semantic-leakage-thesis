"""Compare Target–Neutral stereotype-rate differences across response modes.

Run from repository root: python analysis/26_response_mode_comparison.py
Dependencies: pandas, altair, vl-convert-python
"""
from pathlib import Path
import argparse
import pandas as pd
import altair as alt

MODEL_ORDER = ["OpenAI", "Mistral", "Qwen", "Grok"]
MODEL_NAMES = {"openai": "OpenAI", "mistral": "Mistral", "qwen": "Qwen", "grok": "Grok"}
LANGUAGE_ORDER = ["HU", "EN"]
COLORS = ["#477050", "#3F6FA6"]
MODE_NAMES = {"mc": "MC", "mc_unknown": "MC+NAA"}


def prepare_data(df):
    required = {"model", "language", "response_mode", "target_n", "neutral_n",
                "target_stereotype_n", "neutral_stereotype_n"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.loc[df.response_mode.isin(MODE_NAMES)].copy()
    if len(df) != 16 or df.duplicated(["model", "language", "response_mode"]).any():
        raise ValueError("Expected 16 unique model-language-response_mode rows")
    df["model"] = df["model"].map(MODEL_NAMES)
    df["mode"] = df["response_mode"].map(MODE_NAMES)
    if df["model"].isna().any() or set(df.language) != set(LANGUAGE_ORDER):
        raise ValueError("Unexpected model or language")
    for col in ["target_n", "neutral_n", "target_stereotype_n", "neutral_stereotype_n"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if (df[["target_n", "neutral_n"]] <= 0).any().any():
        raise ValueError("Nonpositive response count")
    df["delta_pp"] = (100 * df.target_stereotype_n / df.target_n
                      - 100 * df.neutral_stereotype_n / df.neutral_n)
    return df


def build_chart(df):
    def panel(model):
        source = df.loc[df.model.eq(model)]
        base = alt.Chart(source).encode(
            x=alt.X("mode:N", sort=["MC", "MC+NAA"], title=None,
                    axis=alt.Axis(labelAngle=0, labelFontSize=14)),
            xOffset=alt.XOffset("language:N", sort=LANGUAGE_ORDER),
            y=alt.Y("delta_pp:Q", title="Target–Neutral különbség (százalékpont)",
                    scale=alt.Scale(domain=[0, 25]),
                    axis=alt.Axis(values=[0, 5, 10, 15, 20, 25], labelFontSize=12, titleFontSize=13)),
            color=alt.Color("language:N", sort=LANGUAGE_ORDER,
                            scale=alt.Scale(domain=LANGUAGE_ORDER, range=COLORS),
                            legend=alt.Legend(title="Nyelv", orient="bottom", labelFontSize=13)),
            tooltip=[alt.Tooltip("model:N", title="Modell"),
                     alt.Tooltip("language:N", title="Nyelv"),
                     alt.Tooltip("mode:N", title="Válaszmód"),
                     alt.Tooltip("delta_pp:Q", title="Különbség (pp)", format=".2f")]
        )
        bars = base.mark_bar(size=40)
        labels = base.mark_text(dy=-9, fontSize=12).encode(
            text=alt.Text("delta_pp:Q", format=".1f"), color=alt.value("#333333"))
        return (bars + labels).properties(width=330, height=250, title=model)

    chart = alt.vconcat(
        alt.hconcat(panel("OpenAI"), panel("Mistral"), spacing=65),
        alt.hconcat(panel("Qwen"), panel("Grok"), spacing=65),
        spacing=65,
    ).resolve_scale(color="shared").properties(
        title="A Target–Neutral különbség válaszmódonként (MC és MC+NAA)"
    )
    return (chart.configure_view(stroke=None)
            .configure_axis(labelFont="Arial", titleFont="Arial", gridColor="#E6E8EB", domain=False)
            .configure_title(font="Arial", fontSize=17, anchor="middle")
            .configure_legend(labelFont="Arial", titleFont="Arial"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("results/analysis/descriptive/02_target_neutral.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/analysis/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(pd.read_csv(args.input, encoding="utf-8-sig"))
    chart = build_chart(data)
    for ext in ("svg", "png", "html"):
        path = args.output_dir / f"07_response_mode_comparison.{ext}"
        chart.save(str(path))
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
