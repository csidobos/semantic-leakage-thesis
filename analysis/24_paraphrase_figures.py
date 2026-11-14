"""Plot Target–Neutral stereotype-rate differences by paraphrase (MC only).

Run from repository root: python analysis/24_paraphrase_figures.py
Dependencies: pandas, altair, vl-convert-python
"""
from pathlib import Path
import argparse
import pandas as pd
import altair as alt

MODEL_ORDER = ["OpenAI", "Mistral", "Qwen", "Grok"]
MODEL_NAMES = {"openai": "OpenAI", "mistral": "Mistral", "qwen": "Qwen", "grok": "Grok"}
LANGUAGE_ORDER = ["HU", "EN"]
LANGUAGE_COLORS = ["#477050", "#3F6FA6"]
PARAPHRASE_ORDER = ["p0", "p1", "p2"]


def make_figure(data: pd.DataFrame) -> alt.Chart:
    required = {"model", "language", "response_mode", "paraphrase", "delta_stereotype_pct_pp_vs_neutral"}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = data.loc[data["response_mode"].eq("mc")].copy()
    frame["model"] = frame["model"].map(MODEL_NAMES)
    frame["delta"] = pd.to_numeric(frame["delta_stereotype_pct_pp_vs_neutral"], errors="raise")
    frame = frame.loc[frame["model"].isin(MODEL_ORDER) & frame["language"].isin(LANGUAGE_ORDER)]
    expected = len(MODEL_ORDER) * len(LANGUAGE_ORDER) * len(PARAPHRASE_ORDER)
    if len(frame) != expected or frame.duplicated(["model", "language", "paraphrase"]).any():
        raise ValueError(f"Expected {expected} unique rows, found {len(frame)}")
    if set(frame["paraphrase"]) != set(PARAPHRASE_ORDER):
        raise ValueError("Expected paraphrase levels p0, p1, p2")

    def panel(model: str) -> alt.Chart:
        subset = frame.loc[frame["model"].eq(model)]
        base = alt.Chart(subset).encode(
            x=alt.X("paraphrase:N", sort=PARAPHRASE_ORDER, title="Parafrázis",
                    axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=14)),
            y=alt.Y("delta:Q", title="Target–Neutral különbség (százalékpont)",
                    scale=alt.Scale(domain=[0, 30]),
                    axis=alt.Axis(values=[0, 5, 10, 15, 20, 25, 30], labelFontSize=12, titleFontSize=13)),
            color=alt.Color("language:N", sort=LANGUAGE_ORDER,
                            scale=alt.Scale(domain=LANGUAGE_ORDER, range=LANGUAGE_COLORS),
                            legend=alt.Legend(title="Nyelv", labelExpr="datum.label == 'HU' ? 'Magyar' : 'Angol'",
                                              labelFontSize=12, titleFontSize=13)),
            detail="language:N",
            tooltip=[alt.Tooltip("model:N", title="Modell"),
                     alt.Tooltip("language:N", title="Nyelv"),
                     alt.Tooltip("paraphrase:N", title="Parafrázis"),
                     alt.Tooltip("delta:Q", title="Különbség (pp)", format="+.2f")],
        )
        return (base.mark_line(strokeWidth=3) + base.mark_point(filled=True, size=110)).properties(
            width=360, height=245, title=model
        )

    top = alt.hconcat(panel("OpenAI"), panel("Mistral"), spacing=38)
    bottom = alt.hconcat(panel("Qwen"), panel("Grok"), spacing=38)
    return (alt.vconcat(top, bottom, spacing=48)
            .resolve_scale(color="shared")
            .configure_view(stroke=None)
            .configure_axis(labelFont="Arial", titleFont="Arial", gridColor="#E6E8EB", domain=False)
            .configure_title(font="Arial", fontSize=18, anchor="middle")
            .configure_legend(labelFont="Arial", titleFont="Arial"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/analysis/descriptive/07_paraphrase_delta.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart = make_figure(pd.read_csv(args.input, encoding="utf-8-sig"))
    for extension in ("svg", "png", "html"):
        output = args.output_dir / f"05_mc_paraphrase_delta.{extension}"
        chart.save(str(output))
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
