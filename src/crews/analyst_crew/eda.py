"""Deterministic exploratory data analysis: computes statistics and renders eda_report.html."""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

TARGET = "Churn"
ID_COL = "CustomerID"
CHURN_COLORS = {0: "#2f6bff", 1: "#d64545"}


def compute_stats(df: pd.DataFrame) -> dict:
    numeric = [c for c in df.select_dtypes("number").columns if c not in (ID_COL, TARGET)]
    categorical = list(df.select_dtypes(include=["object", "string"]).columns)
    # Low-cardinality numeric columns are segments too (CityTier, SatisfactionScore, Complain).
    segments = categorical + [c for c in numeric if df[c].nunique() <= 6]

    churn_rate = float(df[TARGET].mean())
    corr = df[numeric + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    means_by_churn = df.groupby(TARGET)[numeric].mean().T.rename(columns={0: "stayed", 1: "churned"})

    return {
        "rows": len(df),
        "overall_churn_rate": round(churn_rate, 4),
        "churned_customers": int(df[TARGET].sum()),
        "describe": df[numeric].describe().round(2).to_dict(),
        "correlation_with_churn": corr.round(3).to_dict(),
        "mean_by_churn": means_by_churn.round(2).to_dict(orient="index"),
        # Spelled out so an LLM reading the stats cannot flip the direction of a finding.
        # Segment columns (flags, tiers, scores) are left out: their means read as nonsense
        # ("0.53 complaints") and churn_rate_by_segment already covers them.
        "churned_vs_stayed": [
            f"{col}: churned customers average {row.churned:g} vs {row.stayed:g} for those who "
            f"stayed — {'HIGHER' if row.churned > row.stayed else 'LOWER'} among churners"
            for col, row in means_by_churn.round(2).iterrows()
            if row.churned != row.stayed and col not in segments
        ],
        "churn_rate_by_segment": {
            col: df.groupby(col)[TARGET].agg(["mean", "size"]).round(4)
                 .rename(columns={"mean": "churn_rate", "size": "customers"})
                 .to_dict(orient="index")
            for col in segments
        },
        "tenure_bucket_churn": df.assign(
            bucket=pd.cut(df["Tenure"], [-1, 1, 6, 12, 24, 1000],
                          labels=["0-1", "2-6", "7-12", "13-24", "25+"])
        ).groupby("bucket", observed=True)[TARGET].mean().round(4).to_dict(),
    }


def _fig_to_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _charts(df: pd.DataFrame, stats: dict) -> list[tuple[str, str]]:
    sns.set_theme(style="whitegrid")
    charts = []

    fig, ax = plt.subplots(figsize=(5, 3.2))
    counts = df[TARGET].value_counts().sort_index()
    ax.bar(["Stayed (0)", "Churned (1)"], counts.values, color=[CHURN_COLORS[0], CHURN_COLORS[1]])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,} ({v / len(df):.1%})", ha="center", va="bottom")
    ax.set_title("Churn distribution")
    charts.append(("Target distribution", _fig_to_img(fig)))

    corr = pd.Series(stats["correlation_with_churn"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    corr.sort_values().plot.barh(ax=ax, color=["#d64545" if v > 0 else "#2f6bff" for v in corr.sort_values()])
    ax.set_title("Correlation of numeric features with churn")
    ax.axvline(0, color="#333", lw=0.8)
    charts.append(("Correlation with churn", _fig_to_img(fig)))

    tb = pd.Series(stats["tenure_bucket_churn"])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(tb.index.astype(str), tb.values, color="#d64545")
    ax.set_title("Churn rate by tenure (months)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    charts.append(("Churn by tenure", _fig_to_img(fig)))

    segs = list(stats["churn_rate_by_segment"])
    ncols = 3
    nrows = -(-len(segs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.6 * nrows))
    for ax, col in zip(axes.flat, segs):
        s = pd.Series({k: v["churn_rate"] for k, v in stats["churn_rate_by_segment"][col].items()})
        ax.bar([str(k) for k in s.index], s.values, color="#7c5cff")
        ax.axhline(stats["overall_churn_rate"], color="#333", ls="--", lw=1)
        ax.set_title(col)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.tick_params(axis="x", rotation=30)
    for ax in list(axes.flat)[len(segs):]:
        ax.axis("off")
    fig.tight_layout()
    charts.append(("Churn rate by segment (dashed line = overall rate)", _fig_to_img(fig)))

    # Low-cardinality columns already appear in the segment chart above.
    numeric = [c for c in stats["describe"] if c not in stats["churn_rate_by_segment"]]
    if not numeric:
        return charts
    ncols = 4
    nrows = -(-len(numeric) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2 * nrows))
    # Plain matplotlib, not sns.boxplot: seaborn passes a deprecated `vert` argument, and
    # CrewAI's patched warnings.warn crashes on the resulting matplotlib warning.
    for ax, col in zip(axes.flat, numeric):
        groups = [df.loc[df[TARGET] == k, col] for k in (0, 1)]
        box = ax.boxplot(groups, tick_labels=["0", "1"], patch_artist=True, widths=0.6)
        for patch, k in zip(box["boxes"], (0, 1)):
            patch.set_facecolor(CHURN_COLORS[k])
            patch.set_alpha(0.7)
        ax.set_title(col)
    for ax in list(axes.flat)[len(numeric):]:
        ax.axis("off")
    fig.tight_layout()
    charts.append(("Numeric features: stayed (0) vs churned (1)", _fig_to_img(fig)))
    return charts


def _table(d: dict) -> str:
    return pd.DataFrame(d).to_html(classes="tbl", border=0)


def render_report(df: pd.DataFrame, stats: dict, cleaning: dict | None = None) -> str:
    charts = "".join(f"<h2>{html.escape(t)}</h2>{img}" for t, img in _charts(df, stats))
    cleaning_html = ""
    if cleaning:
        items = "".join(f"<li>{html.escape(x)}</li>" for x in cleaning["cleaning_log"])
        cleaning_html = (f"<h2>Cleaning summary</h2><p>Raw rows: {cleaning['before']['rows']:,} → "
                         f"clean rows: {cleaning['after']['rows']:,}</p><ul>{items}</ul>")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>EDA Report — E-commerce Churn</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:0 auto;padding:24px;color:#1a2230}}
h1{{color:#1e3a8a}} h2{{color:#1e3a8a;margin-top:36px;border-bottom:1px solid #dde5f0;padding-bottom:4px}}
.kpis{{display:flex;gap:16px;flex-wrap:wrap}} .kpi{{background:#f4f7fb;border:1px solid #dde5f0;border-radius:10px;padding:12px 18px}}
.kpi b{{display:block;font-size:24px}} img{{max-width:100%}}
.tbl{{border-collapse:collapse;font-size:13px}} .tbl th,.tbl td{{padding:4px 10px;border-bottom:1px solid #dde5f0;text-align:right}}
</style></head><body>
<h1>Exploratory Data Analysis — E-commerce Customer Churn</h1>
<div class="kpis">
 <div class="kpi"><b>{stats['rows']:,}</b>customers</div>
 <div class="kpi"><b>{stats['overall_churn_rate']:.1%}</b>churn rate</div>
 <div class="kpi"><b>{stats['churned_customers']:,}</b>churned customers</div>
</div>
{cleaning_html}
{charts}
<h2>Descriptive statistics</h2>{_table(stats['describe'])}
<h2>Mean by churn status</h2>{pd.DataFrame(stats['mean_by_churn']).T.to_html(classes="tbl", border=0)}
</body></html>"""


def run_eda(clean_path: str | Path, report_path: str | Path, stats_path: str | Path,
            cleaning: dict | None = None) -> dict:
    df = pd.read_csv(clean_path)
    stats = compute_stats(df)
    Path(report_path).write_text(render_report(df, stats, cleaning))
    Path(stats_path).write_text(json.dumps(stats, indent=2, default=str))
    return stats
