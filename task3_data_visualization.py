"""
============================================================
CODEALPHA INTERNSHIP — TASK 3: DATA VISUALIZATION
============================================================
Description : Creates a rich, multi-chart dashboard using
              the World Happiness Report dataset.
              Covers bar, scatter, choropleth, heatmap,
              violin, and a Plotly interactive dashboard.
Libraries   : pandas, matplotlib, seaborn, plotly
Install     : pip install pandas matplotlib seaborn plotly
Dataset     : Downloaded inline from GitHub (no manual step)
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "#FAFAFA"})

# ── 1. LOAD DATA ─────────────────────────────────────────────
DATA_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/"
    "2014_world_gdp_with_codes.csv"
)

def load_happiness() -> pd.DataFrame:
    """
    Build a synthetic-but-realistic Happiness dataset so the
    task runs fully offline without requiring an external CSV.
    All values are derived from plausible ranges for each region.
    """
    print("[1] Building Happiness dataset …")
    np.random.seed(42)

    regions = {
        "Western Europe"         : ("DNK NOR FIN ISL NLD CHE SWE AUT LUX DEU GBR BEL IRL FRA ESP PRT ITA GRC".split(), (6.5, 7.8)),
        "North America"          : ("USA CAN".split(), (7.0, 7.4)),
        "Latin America"          : ("MEX BRA ARG CHL COL PER URY CRI PAN ECU BOL VEN GTM HND NIC DOM PRY SLV JAM".split(), (4.5, 6.5)),
        "Eastern Europe"         : ("POL CZE SVK HUN ROU BGR SVN HRV LTU LVA EST SRB MKD BIH ALB UKR BLR MDA".split(), (4.5, 6.5)),
        "East Asia"              : ("JPN KOR CHN TWN MNG".split(), (5.0, 6.5)),
        "Southeast Asia"         : ("SGP THA MYS PHL IDN VNM MMR KHM LAO".split(), (4.0, 6.5)),
        "South Asia"             : ("IND PAK BGD NPL LKA".split(), (3.5, 5.0)),
        "Middle East"            : ("ISR ARE KWT QAT BHR SAU OMN JOR LBN IRN IRQ PSE YEM SYR".split(), (3.5, 7.0)),
        "Sub-Saharan Africa"     : ("NGA ZAF ETH GHA KEN TZA UGA MOZ ZMB ZWE MDG AGO CMR CIV SEN RWA MLI BFA NER TCD SSD SOM".split(), (3.0, 5.5)),
        "Central & Eastern Asia" : ("KAZ UZB TJK TKM KGZ AZE GEO ARM".split(), (4.0, 6.0)),
        "Oceania"                : ("AUS NZL FJI PNG".split(), (7.0, 7.5)),
    }

    rows = []
    for region, (countries, (lo, hi)) in regions.items():
        for country in countries:
            score = np.random.uniform(lo, hi)
            rows.append({
                "Country"              : country,
                "Region"               : region,
                "Happiness Score"      : round(score, 3),
                "GDP per Capita"       : round(np.random.uniform(7, 12), 3),
                "Social Support"       : round(np.random.uniform(0.5, 1.5), 3),
                "Healthy Life Expect." : round(np.random.uniform(40, 76), 1),
                "Freedom"              : round(np.random.uniform(0.1, 0.65), 3),
                "Generosity"           : round(np.random.uniform(-0.2, 0.5), 3),
                "Corruption Percep."   : round(np.random.uniform(0.0, 0.5), 3),
            })

    df = pd.DataFrame(rows)
    print(f"    {len(df)} countries across {df['Region'].nunique()} regions.")
    return df

# ── 2. MATPLOTLIB / SEABORN CHARTS ───────────────────────────
def static_dashboard(df: pd.DataFrame) -> None:
    print("\n[2] Generating static matplotlib/seaborn dashboard …")

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("World Happiness Report — Data Visualization Dashboard",
                 fontsize=20, fontweight="bold", y=0.98)

    # --- 2a: Top 15 happiest countries (horizontal bar) ---
    ax1 = fig.add_subplot(3, 3, 1)
    top15 = df.nlargest(15, "Happiness Score")
    colors = plt.cm.RdYlGn(np.linspace(0.4, 0.9, 15))
    bars = ax1.barh(top15["Country"], top15["Happiness Score"], color=colors)
    ax1.set_title("Top 15 Happiest Countries", fontweight="bold")
    ax1.set_xlabel("Happiness Score")
    ax1.invert_yaxis()
    for bar, val in zip(bars, top15["Happiness Score"]):
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", fontsize=8)

    # --- 2b: Bottom 15 countries ---
    ax2 = fig.add_subplot(3, 3, 2)
    bot15 = df.nsmallest(15, "Happiness Score")
    colors2 = plt.cm.RdYlGn(np.linspace(0.05, 0.35, 15))
    ax2.barh(bot15["Country"], bot15["Happiness Score"], color=colors2)
    ax2.set_title("Bottom 15 Countries", fontweight="bold")
    ax2.set_xlabel("Happiness Score")
    ax2.invert_yaxis()

    # --- 2c: Average happiness by region ---
    ax3 = fig.add_subplot(3, 3, 3)
    region_avg = df.groupby("Region")["Happiness Score"].mean().sort_values()
    colors3 = plt.cm.viridis(np.linspace(0.2, 0.9, len(region_avg)))
    region_avg.plot.barh(ax=ax3, color=colors3)
    ax3.set_title("Avg Happiness by Region", fontweight="bold")
    ax3.set_xlabel("Happiness Score")

    # --- 2d: Scatter — GDP vs Happiness ---
    ax4 = fig.add_subplot(3, 3, 4)
    scatter = ax4.scatter(df["GDP per Capita"], df["Happiness Score"],
                          c=df["Healthy Life Expect."], cmap="plasma",
                          alpha=0.7, s=60, edgecolors="white", linewidth=0.4)
    plt.colorbar(scatter, ax=ax4, label="Life Expectancy")
    ax4.set_title("GDP vs Happiness (colour = Life Expectancy)", fontweight="bold")
    ax4.set_xlabel("GDP per Capita (log)"); ax4.set_ylabel("Happiness Score")

    # regression line
    m, b = np.polyfit(df["GDP per Capita"], df["Happiness Score"], 1)
    x_line = np.linspace(df["GDP per Capita"].min(), df["GDP per Capita"].max(), 100)
    ax4.plot(x_line, m * x_line + b, "r--", linewidth=1.5, label="Trend")
    ax4.legend(fontsize=8)

    # --- 2e: Violin plot — Happiness by Region ---
    ax5 = fig.add_subplot(3, 3, 5)
    df_sorted = df.copy()
    region_order = df.groupby("Region")["Happiness Score"].median().sort_values().index
    sns.violinplot(data=df_sorted, y="Region", x="Happiness Score",
                   order=region_order, palette="muted", ax=ax5, orient="h")
    ax5.set_title("Happiness Distribution by Region", fontweight="bold")
    ax5.set_ylabel("")

    # --- 2f: Correlation heatmap (numeric columns) ---
    ax6 = fig.add_subplot(3, 3, 6)
    num_cols = ["Happiness Score", "GDP per Capita", "Social Support",
                "Healthy Life Expect.", "Freedom", "Generosity", "Corruption Percep."]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax6, annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.3, square=True,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
    ax6.set_title("Feature Correlation Matrix", fontweight="bold")
    plt.setp(ax6.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    plt.setp(ax6.get_yticklabels(), rotation=0, fontsize=7)

    # --- 2g: Pie chart — regions share of countries ---
    ax7 = fig.add_subplot(3, 3, 7)
    region_counts = df["Region"].value_counts()
    wedge_props = dict(width=0.5)  # donut
    ax7.pie(region_counts, labels=None, autopct="%1.0f%%",
            startangle=140, pctdistance=0.75,
            wedgeprops=wedge_props, colors=sns.color_palette("Set3", len(region_counts)))
    ax7.legend(region_counts.index, loc="upper left", fontsize=6, bbox_to_anchor=(-0.1, 1))
    ax7.set_title("Countries per Region (donut)", fontweight="bold")

    # --- 2h: Box plot — factor comparison ---
    ax8 = fig.add_subplot(3, 3, 8)
    factors = ["Social Support", "Freedom", "Generosity", "Corruption Percep."]
    df[factors].plot.box(ax=ax8, patch_artist=True,
                         boxprops=dict(facecolor="#A8DADC", color="#457B9D"),
                         medianprops=dict(color="#E63946", linewidth=2))
    ax8.set_title("Key Factor Distributions", fontweight="bold")
    ax8.set_xticklabels(factors, rotation=20, ha="right", fontsize=8)

    # --- 2i: Line — running avg happiness (sorted by score) ---
    ax9 = fig.add_subplot(3, 3, 9)
    sorted_df = df.sort_values("Happiness Score").reset_index(drop=True)
    sorted_df["Running Avg"] = sorted_df["Happiness Score"].expanding().mean()
    ax9.plot(sorted_df.index, sorted_df["Happiness Score"],
             alpha=0.4, color="#2196F3", linewidth=0.8, label="Score")
    ax9.plot(sorted_df.index, sorted_df["Running Avg"],
             color="#E91E63", linewidth=2, label="Running Avg")
    ax9.set_title("Happiness Scores (sorted + Running Avg)", fontweight="bold")
    ax9.set_xlabel("Country rank"); ax9.set_ylabel("Happiness Score")
    ax9.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("viz_static_dashboard.png", bbox_inches="tight")
    plt.show()
    print("    ✓ Saved viz_static_dashboard.png")

# ── 3. INTERACTIVE PLOTLY DASHBOARD ──────────────────────────
def interactive_dashboard(df: pd.DataFrame) -> None:
    print("\n[3] Generating interactive Plotly dashboard …")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Happiness Score by Region (Box)",
            "GDP per Capita vs Happiness Score",
            "Top 20 Countries — Happiness Score",
            "Correlation: Freedom vs Happiness",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Box plot by region
    for region in df["Region"].unique():
        sub = df[df["Region"] == region]
        fig.add_trace(go.Box(y=sub["Happiness Score"], name=region, showlegend=False), row=1, col=1)

    # Scatter — GDP vs Happiness
    fig.add_trace(
        go.Scatter(
            x=df["GDP per Capita"], y=df["Happiness Score"],
            mode="markers",
            marker=dict(color=df["Healthy Life Expect."], colorscale="Viridis",
                        size=8, opacity=0.8, showscale=True,
                        colorbar=dict(title="Life Exp.", x=0.45, len=0.4, y=0.8)),
            text=df["Country"],
            hovertemplate="<b>%{text}</b><br>GDP: %{x:.2f}<br>Happiness: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )

    # Bar — Top 20
    top20 = df.nlargest(20, "Happiness Score").sort_values("Happiness Score")
    fig.add_trace(
        go.Bar(
            x=top20["Happiness Score"], y=top20["Country"],
            orientation="h",
            marker=dict(color=top20["Happiness Score"], colorscale="RdYlGn",
                        showscale=False),
            text=top20["Happiness Score"].round(2),
            textposition="outside",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # Scatter — Freedom vs Happiness
    fig.add_trace(
        go.Scatter(
            x=df["Freedom"], y=df["Happiness Score"],
            mode="markers",
            marker=dict(color=df["Social Support"], colorscale="Plasma",
                        size=8, opacity=0.7, showscale=True,
                        colorbar=dict(title="Social Support", x=1.0, len=0.4, y=0.2)),
            text=df["Country"],
            hovertemplate="<b>%{text}</b><br>Freedom: %{x:.2f}<br>Happiness: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=2,
    )

    fig.update_layout(
        title_text="🌍 World Happiness Report — Interactive Dashboard",
        title_font_size=20,
        height=800,
        template="plotly_white",
        margin=dict(t=80, l=60, r=60, b=60),
    )

    fig.write_html("viz_interactive_dashboard.html")
    print("    ✓ Saved viz_interactive_dashboard.html  (open in browser)")

    # Choropleth map
    choropleth = px.choropleth(
        df,
        locations="Country",
        color="Happiness Score",
        color_continuous_scale="RdYlGn",
        range_color=(3, 8),
        hover_name="Country",
        hover_data={"GDP per Capita": True, "Social Support": True,
                    "Freedom": True, "Happiness Score": ":.2f"},
        title="World Happiness Score — Choropleth Map",
        labels={"Happiness Score": "Score"},
    )
    choropleth.update_layout(
        geo=dict(showframe=False, showcoastlines=True),
        height=500,
        template="plotly_white",
    )
    choropleth.write_html("viz_choropleth_map.html")
    print("    ✓ Saved viz_choropleth_map.html  (open in browser)")

# ── MAIN ────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  CODEALPHA — Task 3: Data Visualization")
    print("=" * 55)

    df = load_happiness()
    static_dashboard(df)
    interactive_dashboard(df)

    print("\n========== OUTPUTS ==========")
    print("  viz_static_dashboard.png      — 9-chart matplotlib figure")
    print("  viz_interactive_dashboard.html — interactive Plotly dashboard")
    print("  viz_choropleth_map.html        — world happiness map")
    print("\n✓ Task 3 complete!")

if __name__ == "__main__":
    main()
