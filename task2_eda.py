"""
============================================================
CODEALPHA INTERNSHIP — TASK 2: EXPLORATORY DATA ANALYSIS
============================================================
Description : Performs full EDA on the Titanic dataset.
              Covers structure inspection, cleaning, stats,
              hypothesis testing, and visualisations.
Libraries   : pandas, numpy, matplotlib, seaborn, scipy
Install     : pip install pandas numpy matplotlib seaborn scipy
Dataset     : Built-in via seaborn (no download needed)
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── 1. LOAD DATA ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print("\n[1] Loading Titanic dataset via seaborn …")
    df = sns.load_dataset("titanic")
    print(f"    Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

# ── 2. STRUCTURE INSPECTION ──────────────────────────────────
def inspect_structure(df: pd.DataFrame) -> None:
    print("\n[2] Dataset Structure")
    print("-" * 40)
    print(df.dtypes)
    print(f"\nFirst 5 rows:\n{df.head()}")

# ── 3. MISSING VALUES ────────────────────────────────────────
def analyse_missing(df: pd.DataFrame) -> None:
    print("\n[3] Missing Values")
    print("-" * 40)
    missing = df.isnull().sum()
    pct     = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"Missing": missing, "Percent (%)": pct})
    print(summary[summary["Missing"] > 0])

# ── 4. CLEAN DATA ────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4] Cleaning Data …")
    df = df.copy()
    df["age"].fillna(df["age"].median(), inplace=True)
    df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)
    df.drop(columns=["deck"], inplace=True, errors="ignore")  # >75% missing
    df["fare"].fillna(df["fare"].median(), inplace=True)
    print(f"    Remaining nulls: {df.isnull().sum().sum()}")
    return df

# ── 5. DESCRIPTIVE STATISTICS ────────────────────────────────
def descriptive_stats(df: pd.DataFrame) -> None:
    print("\n[5] Descriptive Statistics")
    print("-" * 40)
    print(df.describe().round(2))
    print(f"\nSurvival rate: {df['survived'].mean()*100:.1f}%")
    print(f"Gender split : Male={df['sex'].value_counts()['male']}, Female={df['sex'].value_counts()['female']}")

# ── 6. UNIVARIATE ANALYSIS ───────────────────────────────────
def univariate_plots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Univariate Analysis — Titanic", fontsize=16, fontweight="bold")

    # Age distribution
    axes[0, 0].hist(df["age"].dropna(), bins=30, color="#5E81AC", edgecolor="white")
    axes[0, 0].set_title("Age Distribution"); axes[0, 0].set_xlabel("Age")

    # Fare distribution
    axes[0, 1].hist(df["fare"].dropna(), bins=40, color="#A3BE8C", edgecolor="white")
    axes[0, 1].set_title("Fare Distribution"); axes[0, 1].set_xlabel("Fare (£)")

    # Survival count
    df["survived"].value_counts().rename({0: "Died", 1: "Survived"}).plot.bar(
        ax=axes[0, 2], color=["#BF616A", "#5E81AC"], edgecolor="white", rot=0)
    axes[0, 2].set_title("Survival Count")

    # Pclass
    df["pclass"].value_counts().sort_index().plot.bar(
        ax=axes[1, 0], color="#EBCB8B", edgecolor="white", rot=0)
    axes[1, 0].set_title("Passenger Class")

    # Sex
    df["sex"].value_counts().plot.bar(
        ax=axes[1, 1], color=["#81A1C1", "#B48EAD"], edgecolor="white", rot=0)
    axes[1, 1].set_title("Gender Distribution")

    # Embarked
    df["embarked"].value_counts().plot.bar(
        ax=axes[1, 2], color="#88C0D0", edgecolor="white", rot=0)
    axes[1, 2].set_title("Port of Embarkation")

    plt.tight_layout()
    plt.savefig("eda_univariate.png")
    plt.show()
    print("    ✓ Saved eda_univariate.png")

# ── 7. BIVARIATE ANALYSIS ────────────────────────────────────
def bivariate_plots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Bivariate Analysis — Titanic", fontsize=16, fontweight="bold")

    # Survival by Sex
    survival_sex = df.groupby("sex")["survived"].mean().rename({"female": "Female", "male": "Male"})
    survival_sex.plot.bar(ax=axes[0, 0], color=["#B48EAD", "#81A1C1"], edgecolor="white", rot=0)
    axes[0, 0].set_title("Survival Rate by Gender"); axes[0, 0].set_ylabel("Survival Rate")
    axes[0, 0].set_ylim(0, 1)

    # Survival by Class
    survival_class = df.groupby("pclass")["survived"].mean()
    survival_class.plot.bar(ax=axes[0, 1], color=["#88C0D0", "#5E81AC", "#4C566A"], edgecolor="white", rot=0)
    axes[0, 1].set_title("Survival Rate by Class"); axes[0, 1].set_ylabel("Survival Rate")
    axes[0, 1].set_ylim(0, 1)

    # Age vs Fare (coloured by survival)
    for survived, grp in df.groupby("survived"):
        label = "Survived" if survived else "Died"
        color = "#A3BE8C" if survived else "#BF616A"
        axes[1, 0].scatter(grp["age"], grp["fare"], alpha=0.4, s=20, label=label, color=color)
    axes[1, 0].set_title("Age vs Fare (by Survival)")
    axes[1, 0].set_xlabel("Age"); axes[1, 0].set_ylabel("Fare"); axes[1, 0].legend()

    # Age box plot by survival
    df["Survived Label"] = df["survived"].map({0: "Died", 1: "Survived"})
    df.boxplot(column="age", by="Survived Label", ax=axes[1, 1],
               boxprops=dict(color="#5E81AC"), medianprops=dict(color="#BF616A"))
    axes[1, 1].set_title("Age Distribution by Survival")
    axes[1, 1].set_xlabel(""); axes[1, 1].set_ylabel("Age")
    plt.suptitle("")

    plt.tight_layout()
    plt.savefig("eda_bivariate.png")
    plt.show()
    print("    ✓ Saved eda_bivariate.png")

# ── 8. CORRELATION HEATMAP ───────────────────────────────────
def correlation_heatmap(df: pd.DataFrame) -> None:
    num_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = df[num_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("eda_correlation.png")
    plt.show()
    print("    ✓ Saved eda_correlation.png")

# ── 9. HYPOTHESIS TESTS ──────────────────────────────────────
def hypothesis_tests(df: pd.DataFrame) -> None:
    print("\n[9] Hypothesis Testing")
    print("-" * 40)

    # H1: Survival differs by gender (Chi-square)
    ct = pd.crosstab(df["sex"], df["survived"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    print(f"\nH1 — Survival vs Gender (Chi-square)")
    print(f"   χ² = {chi2:.3f},  p-value = {p:.6f}")
    print(f"   → {'REJECT H0 — significant difference' if p < 0.05 else 'Fail to reject H0'}")

    # H2: Age differs between survivors and non-survivors (t-test)
    survived_ages = df[df["survived"] == 1]["age"].dropna()
    died_ages     = df[df["survived"] == 0]["age"].dropna()
    t_stat, p2 = stats.ttest_ind(survived_ages, died_ages)
    print(f"\nH2 — Age vs Survival (Independent t-test)")
    print(f"   t = {t_stat:.3f},  p-value = {p2:.6f}")
    print(f"   → {'REJECT H0 — age significantly differs' if p2 < 0.05 else 'Fail to reject H0'}")

    # H3: Fare differs across classes (ANOVA)
    groups  = [df[df["pclass"] == c]["fare"].dropna() for c in [1, 2, 3]]
    f_stat, p3 = stats.f_oneway(*groups)
    print(f"\nH3 — Fare vs Class (One-way ANOVA)")
    print(f"   F = {f_stat:.3f},  p-value = {p3:.6f}")
    print(f"   → {'REJECT H0 — fare differs across classes' if p3 < 0.05 else 'Fail to reject H0'}")

# ── MAIN ────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  CODEALPHA — Task 2: Exploratory Data Analysis")
    print("=" * 55)

    df = load_data()
    inspect_structure(df)
    analyse_missing(df)
    df = clean_data(df)
    descriptive_stats(df)

    print("\n[6] Generating Univariate Plots …")
    univariate_plots(df)

    print("\n[7] Generating Bivariate Plots …")
    bivariate_plots(df)

    print("\n[8] Generating Correlation Heatmap …")
    correlation_heatmap(df)

    hypothesis_tests(df)

    print("\n✓ Task 2 complete!  All plots saved as PNG files.")

if __name__ == "__main__":
    main()
