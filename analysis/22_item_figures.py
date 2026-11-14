"""Plot item-level Target–Neutral differences for MC responses.

Run from repository root: python analysis/22_item_figures.py
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


def make_figures(data: pd.DataFrame) -> list[alt.Chart]:
    required = {"model", "language", "response_mode", "item_id", "delta_stereotype_pct_pp_vs_neutral"}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = data.loc[data["response_mode"].eq("mc")].copy()
    frame["model"] = frame["model"].map(MODEL_NAMES)
    frame["delta"] = pd.to_numeric(frame["delta_stereotype_pct_pp_vs_neutral"], errors="raise")
    frame = frame.loc[frame["model"].isin(MODEL_ORDER) & frame["language"].isin(LANGUAGE_ORDER)]
    item_order = sorted(frame["item_id"].unique())
    expected = len(MODEL_ORDER) * len(LANGUAGE_ORDER) * 32
    if len(frame) != expected or len(item_order) != 32 or frame.duplicated(["model", "language", "item_id"]).any():
        raise ValueError(f"Expected {expected} unique model-language-item rows, found {len(frame)}")

    # All panels use a common symmetric scale for direct comparison.
    bound = max(10, int((frame["delta"].abs().max() + 9) // 10) * 10)

    def panel(model: str) -> alt.Chart:
        subset = frame.loc[frame["model"].eq(model)]
        zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color="#777777", strokeWidth=1).encode(x="zero:Q")
        dots = (
            alt.Chart(subset).mark_circle(size=95, opacity=0.85)
            .encode(
                y=alt.Y("item_id:N", sort=item_order, title=None,
                        axis=alt.Axis(labelFontSize=14, labelLimit=180, labelOverlap=False)),
                x=alt.X("delta:Q", title="Target–Neutral különbség (százalékpont)",
                        scale=alt.Scale(domain=[-bound, bound]),
                        axis=alt.Axis(grid=True, tickCount=7, labelFontSize=13, titleFontSize=14, titlePadding=12)),
                color=alt.Color("language:N", sort=LANGUAGE_ORDER,
                                scale=alt.Scale(domain=LANGUAGE_ORDER, range=LANGUAGE_COLORS),
                                legend=alt.Legend(title="Nyelv", labelFontSize=13, titleFontSize=14, labelExpr="datum.label == 'HU' ? 'Magyar' : 'Angol'")),
                tooltip=[alt.Tooltip("item_id:N", title="Item"),
                         alt.Tooltip("language:N", title="Nyelv"),
                         alt.Tooltip("delta:Q", title="Különbség (pp)", format="+.1f")],
            )
        )
        return (zero + dots).properties(width=440, height=950, title=model)

    # Two A4-friendly figures instead of squeezing four models onto one page.
    figures = []
    for left, right in (("OpenAI", "Mistral"), ("Qwen", "Grok")):
        chart = alt.hconcat(panel(left), panel(right), spacing=55)
        chart = (chart.configure_view(stroke=None)
                 .configure_axis(labelFont="Arial", titleFont="Arial", gridColor="#E6E8EB", domain=False)
                 .configure_title(font="Arial", fontSize=19, anchor="middle")
                 .configure_legend(labelFont="Arial", titleFont="Arial"))
        figures.append(chart)
    return figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/analysis/descriptive/04_item_delta.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = make_figures(pd.read_csv(args.input, encoding="utf-8-sig"))
    for page, chart in enumerate(figures, start=1):
        for extension in ("svg", "png", "html"):
            output = args.output_dir / f"03_mc_item_delta_page{page}.{extension}"
            chart.save(str(output))
            print(f"Saved: {output}")


if __name__ == "__main__":
    main()
