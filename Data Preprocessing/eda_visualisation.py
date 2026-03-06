"""
eda_visualisation.py
====================
Exploratory Data Analysis of the cleaned HDB Resale dataset.
Loads data from hdb_resale.db (produced by data_pipeline.py).

Purpose: understand the raw cleaned data BEFORE any feature engineering
so we can make informed decisions about:
  - Which features to include in the ML model
  - Whether log transforms are needed (skewness)
  - How strongly each feature correlates with price
  - Where outliers are and whether capping is justified

Outputs 17 plots to eda_plots/.

Run:
    python eda_visualisation.py
"""

import os
import sqlite3

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH    = "hdb_resale.db"
TABLE_NAME = "resale_prices"
OUTPUT_DIR = "eda_plots"

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUMERIC_FEATURES = [
    "floor_area_sqm",
    "storey_midpoint",
    "lease_commence_date",
    "remaining_lease",       # years
    "remaining_lease_months",
    "resale_price",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def savefig(name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def price_fmt(x, _):
    return f"{x/1e3:.0f}k"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    print(f"Loading data from {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()

    for col in NUMERIC_FEATURES + ["year", "month_num"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["town", "flat_type", "flat_model"]:
        df[col] = df[col].astype(str).str.strip()

    print(f"  {len(df):,} rows loaded.\n")
    return df


# ---------------------------------------------------------------------------
# 01 — Missing values
# ---------------------------------------------------------------------------

def plot_missing_values(df: pd.DataFrame) -> None:
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print("  No missing values — skipping plot 01.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    missing.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Missing Values by Column (%)")
    ax.set_ylabel("% Missing")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    savefig("01_missing_values.png")


# ---------------------------------------------------------------------------
# 02 — Price distribution
# ---------------------------------------------------------------------------

def plot_price_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    price = df["resale_price"].dropna()

    axes[0].hist(price / 1e3, bins=80, color="steelblue", edgecolor="white")
    axes[0].set_title("Resale Price Distribution")
    axes[0].set_xlabel("Price (SGD thousands)")
    axes[0].set_ylabel("Count")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(price_fmt))

    axes[1].hist(price / 1e3, bins=80, color="teal", edgecolor="white", log=True)
    axes[1].set_title("Resale Price (log-scale count)")
    axes[1].set_xlabel("Price (SGD thousands)")
    axes[1].set_ylabel("Count (log)")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(price_fmt))

    savefig("02_price_distribution.png")


# ---------------------------------------------------------------------------
# 03 — Median price over time (quarterly)
# ---------------------------------------------------------------------------

def plot_price_over_time(df: pd.DataFrame) -> None:
    df2 = df.copy()
    df2["period"] = pd.to_datetime(df2["month"], format="%Y-%m", errors="coerce").dt.to_period("Q")
    monthly = df2.groupby("period")["resale_price"].median().reset_index()
    monthly["period"] = monthly["period"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly["period"], monthly["resale_price"] / 1e3, linewidth=1.8, color="steelblue")
    ax.fill_between(monthly["period"], monthly["resale_price"] / 1e3, alpha=0.15, color="steelblue")
    ax.set_title("Median Resale Price Over Time (Quarterly)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Median Price (SGD thousands)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(price_fmt))
    savefig("03_price_over_time.png")


# ---------------------------------------------------------------------------
# 04 — Transaction volume per year
# ---------------------------------------------------------------------------

def plot_volume_over_time(df: pd.DataFrame) -> None:
    yearly = df.groupby("year").size().reset_index(name="count")
    yearly = yearly[yearly["year"].notna()]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(yearly["year"].astype(int), yearly["count"], color="mediumseagreen", edgecolor="white")
    ax.set_title("Number of Transactions per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Transactions")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    savefig("04_volume_over_time.png")


# ---------------------------------------------------------------------------
# 05 — Price by flat type
# ---------------------------------------------------------------------------

def plot_price_by_flat_type(df: pd.DataFrame) -> None:
    order = df.groupby("flat_type")["resale_price"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="flat_type", y="resale_price", order=order,
                ax=ax, showfliers=False, palette="Blues")
    ax.set_title("Resale Price by Flat Type")
    ax.set_xlabel("Flat Type")
    ax.set_ylabel("Price (SGD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(price_fmt))
    plt.xticks(rotation=30, ha="right")
    savefig("05_price_by_flat_type.png")


# ---------------------------------------------------------------------------
# 06 — Median price by town
# ---------------------------------------------------------------------------

def plot_price_by_town(df: pd.DataFrame) -> None:
    town_median = df.groupby("town")["resale_price"].median().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    town_median.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Median Resale Price by Town")
    ax.set_xlabel("Median Price (SGD)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(price_fmt))
    savefig("06_price_by_town.png")


# ---------------------------------------------------------------------------
# 07 — Price vs floor area
# ---------------------------------------------------------------------------

def plot_price_vs_floor_area(df: pd.DataFrame) -> None:
    sample = df[["floor_area_sqm", "resale_price"]].dropna().sample(
        min(20_000, len(df)), random_state=42
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(sample["floor_area_sqm"], sample["resale_price"] / 1e3,
                    alpha=0.15, s=8, c=sample["resale_price"] / 1e3, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="Price (SGD thousands)")
    ax.set_title("Resale Price vs Floor Area (20k sample)")
    ax.set_xlabel("Floor Area (sqm)")
    ax.set_ylabel("Price (SGD thousands)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(price_fmt))
    savefig("07_price_vs_floor_area.png")


# ---------------------------------------------------------------------------
# 08 — Median price by storey midpoint
# ---------------------------------------------------------------------------

def plot_price_vs_storey(df: pd.DataFrame) -> None:
    storey_med = (df.groupby("storey_midpoint")["resale_price"]
                  .median().reset_index().dropna().sort_values("storey_midpoint"))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(storey_med["storey_midpoint"], storey_med["resale_price"] / 1e3,
           width=2, color="coral", edgecolor="white")
    ax.set_title("Median Resale Price by Storey Midpoint")
    ax.set_xlabel("Storey Midpoint")
    ax.set_ylabel("Median Price (SGD thousands)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(price_fmt))
    savefig("08_price_vs_storey.png")


# ---------------------------------------------------------------------------
# 09 — Price vs remaining lease (years)
# ---------------------------------------------------------------------------

def plot_price_vs_remaining_lease(df: pd.DataFrame) -> None:
    data = df[["remaining_lease", "resale_price"]].dropna().copy()
    # Bin into 5-year groups
    data["lease_bin"] = (data["remaining_lease"] / 5).round() * 5
    lease_med = (data.groupby("lease_bin")["resale_price"]
                 .median().reset_index().sort_values("lease_bin"))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(lease_med["lease_bin"], lease_med["resale_price"] / 1e3,
            marker="o", linewidth=1.8, color="darkorange")
    ax.set_title("Median Resale Price vs Remaining Lease (5-year bins)")
    ax.set_xlabel("Remaining Lease (years)")
    ax.set_ylabel("Median Price (SGD thousands)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(price_fmt))
    savefig("09_price_vs_remaining_lease.png")


# ---------------------------------------------------------------------------
# 10 — Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = ["resale_price", "floor_area_sqm", "storey_midpoint",
            "remaining_lease", "lease_commence_date", "year"]
    corr = df[cols].dropna().corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    savefig("10_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# 11 — Flat type mix over time (stacked area)
# ---------------------------------------------------------------------------

def plot_flat_type_mix(df: pd.DataFrame) -> None:
    pivot = df.groupby(["year", "flat_type"]).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct = pivot_pct[pivot_pct.index.notna()]

    fig, ax = plt.subplots(figsize=(14, 6))
    pivot_pct.plot(kind="area", stacked=True, ax=ax, alpha=0.8, colormap="tab10")
    ax.set_title("Flat Type Mix Over Years (%)")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of Transactions")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    savefig("11_flat_type_mix_over_time.png")


# ---------------------------------------------------------------------------
# 12 — Price trend by flat type
# ---------------------------------------------------------------------------

def plot_price_trend_by_flat_type(df: pd.DataFrame) -> None:
    top_types = df["flat_type"].value_counts().head(5).index.tolist()
    yearly = (df[df["flat_type"].isin(top_types)]
              .groupby(["year", "flat_type"])["resale_price"]
              .median().reset_index().dropna())

    fig, ax = plt.subplots(figsize=(14, 6))
    for ft in top_types:
        sub = yearly[yearly["flat_type"] == ft]
        ax.plot(sub["year"], sub["resale_price"] / 1e3, marker="o", markersize=3, label=ft)
    ax.set_title("Median Resale Price Trend by Flat Type (Top 5)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Median Price (SGD thousands)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(price_fmt))
    ax.legend()
    savefig("12_price_trend_by_flat_type.png")


# ---------------------------------------------------------------------------
# 13 — Numeric feature distributions (skewness check)
# ---------------------------------------------------------------------------

def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """
    Histogram + KDE for each numeric feature.
    Skewness score printed — helps decide if log transform is needed.
    Rule of thumb: |skew| > 1 = heavily skewed, consider log transform.
    """
    plot_cols = ["resale_price", "floor_area_sqm", "storey_midpoint",
                 "remaining_lease", "lease_commence_date"]
    n = len(plot_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle("Numeric Feature Distributions (skewness shown)", fontsize=13)

    print("\n  Skewness of numeric features:")
    for ax, col in zip(axes, plot_cols):
        data = df[col].dropna()
        skew = stats.skew(data)
        print(f"    {col:<28}: {skew:+.2f}")
        ax.hist(data, bins=60, color="steelblue", edgecolor="white", density=True, alpha=0.7)
        data.plot.kde(ax=ax, color="darkorange", linewidth=1.8)
        ax.set_title(f"{col}\nskew={skew:+.2f}", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("Density")

    savefig("13_numeric_distributions.png")


# ---------------------------------------------------------------------------
# 14 — Feature correlations with resale_price
# ---------------------------------------------------------------------------

def plot_feature_correlations(df: pd.DataFrame) -> None:
    """
    Pearson correlation of each numeric feature with resale_price.
    Higher absolute value = stronger linear relationship with price.
    """
    num_cols = ["floor_area_sqm", "storey_midpoint", "remaining_lease",
                "remaining_lease_months", "lease_commence_date", "year", "month_num"]
    corrs = (df[num_cols + ["resale_price"]].dropna()
             .corr()["resale_price"]
             .drop("resale_price")
             .sort_values())

    colors = ["crimson" if v < 0 else "steelblue" for v in corrs]
    fig, ax = plt.subplots(figsize=(9, 6))
    corrs.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Pearson Correlation of Features with Resale Price")
    ax.set_xlabel("Correlation coefficient")
    for bar, val in zip(ax.patches, corrs):
        ax.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)
    savefig("14_feature_correlations.png")


# ---------------------------------------------------------------------------
# 15 — Log transform check (price + floor area)
# ---------------------------------------------------------------------------

def plot_log_transform_check(df: pd.DataFrame) -> None:
    """
    Side-by-side raw vs log distributions for price and floor_area_sqm.
    If log distribution is more bell-shaped, log transform may improve model.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Log Transform Check — Raw vs Log Distribution", fontsize=13)

    pairs = [
        (df["resale_price"].dropna(),      "Resale Price (SGD)", axes[0][0], axes[0][1]),
        (df["floor_area_sqm"].dropna(),    "Floor Area (sqm)",   axes[1][0], axes[1][1]),
    ]
    for data, label, ax_raw, ax_log in pairs:
        skew_raw = stats.skew(data)
        skew_log = stats.skew(np.log1p(data))

        ax_raw.hist(data, bins=60, color="steelblue", edgecolor="white", density=True, alpha=0.75)
        data.plot.kde(ax=ax_raw, color="darkorange", linewidth=1.5)
        ax_raw.set_title(f"{label}\nRaw  (skew={skew_raw:+.2f})", fontsize=9)

        log_data = np.log1p(data)
        ax_log.hist(log_data, bins=60, color="mediumseagreen", edgecolor="white", density=True, alpha=0.75)
        log_data.plot.kde(ax=ax_log, color="darkorange", linewidth=1.5)
        ax_log.set_title(f"log(1 + {label})\n(skew={skew_log:+.2f})", fontsize=9)

    savefig("15_log_transform_check.png")


# ---------------------------------------------------------------------------
# 16 — Price by flat model
# ---------------------------------------------------------------------------

def plot_price_by_flat_model(df: pd.DataFrame) -> None:
    order = df.groupby("flat_model")["resale_price"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df, x="resale_price", y="flat_model", order=order,
                ax=ax, showfliers=False, palette="coolwarm")
    ax.set_title("Resale Price by Flat Model")
    ax.set_xlabel("Price (SGD)")
    ax.set_ylabel("Flat Model")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(price_fmt))
    savefig("16_price_by_flat_model.png")


# ---------------------------------------------------------------------------
# 17 — Outlier analysis (IQR box plots per flat type)
# ---------------------------------------------------------------------------

def plot_outlier_analysis(df: pd.DataFrame) -> None:
    """
    Box plots with outliers shown (fliers). The spread of dots above the
    whiskers shows how many extreme values exist per flat type — justifies
    IQR-based capping in preprocessing.
    """
    order = df.groupby("flat_type")["resale_price"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(data=df, x="flat_type", y="resale_price", order=order,
                ax=ax, showfliers=True,
                flierprops=dict(marker=".", alpha=0.2, markersize=2),
                palette="Oranges")
    ax.set_title("Resale Price Outliers by Flat Type (IQR whiskers × 1.5)")
    ax.set_xlabel("Flat Type")
    ax.set_ylabel("Price (SGD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(price_fmt))
    plt.xticks(rotation=30, ha="right")
    savefig("17_outlier_analysis.png")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATASET SUMMARY (raw cleaned data from DB)")
    print("=" * 60)
    print(f"  Rows              : {len(df):,}")
    print(f"  Columns           : {df.shape[1]}")
    print(f"  Date range        : {df['month'].min()} → {df['month'].max()}")
    print(f"  Towns             : {df['town'].nunique()} unique")
    print(f"  Flat types        : {sorted(df['flat_type'].dropna().unique())}")
    print(f"\n  Resale Price (SGD):")
    for stat, val in df["resale_price"].describe().items():
        print(f"    {stat:8s}: {val:>14,.0f}")
    print(f"\n  Remaining Lease (years):")
    for stat, val in df["remaining_lease"].describe().items():
        print(f"    {stat:8s}: {val:>10.2f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_data()
    print_summary(df)

    print(f"\nGenerating {OUTPUT_DIR}/ ...")

    # Dataset overview
    plot_missing_values(df)
    plot_price_distribution(df)
    plot_price_over_time(df)
    plot_volume_over_time(df)

    # Feature vs price
    plot_price_by_flat_type(df)
    plot_price_by_town(df)
    plot_price_vs_floor_area(df)
    plot_price_vs_storey(df)
    plot_price_vs_remaining_lease(df)

    # Multicollinearity
    plot_correlation_heatmap(df)

    # Time patterns
    plot_flat_type_mix(df)
    plot_price_trend_by_flat_type(df)

    # Feature selection insights
    plot_numeric_distributions(df)
    plot_feature_correlations(df)
    plot_log_transform_check(df)
    plot_price_by_flat_model(df)
    plot_outlier_analysis(df)

    print(f"\nAll 17 plots saved to '{OUTPUT_DIR}/'.")
    print("\nFeature selection guide:")
    print("  Plot 13 — check skewness: |skew| > 1 → consider log transform")
    print("  Plot 14 — use features with |corr| > 0.2 as strong candidates")
    print("  Plot 15 — log(price) more normal? → use log(resale_price) as target")
    print("  Plot 10 — correlated features → may drop one to reduce multicollinearity")
    print("  Plot 17 — large outlier spread → IQR capping is justified")


if __name__ == "__main__":
    main()
