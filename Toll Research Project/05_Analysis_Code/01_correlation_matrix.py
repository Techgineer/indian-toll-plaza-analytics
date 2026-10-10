from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "04_Working_Dataset" / "working_dataset.xlsx"
OUTPUT_FIGURES_DIR = BASE_DIR / "06_Results" / "figures"
OUTPUT_RESULTS_DIR = BASE_DIR / "06_Results"
OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


corr_columns = [
    "ADT",
    "PHV",
    "PHF",
    "K",
    "D",
    "PCU",
    "HV%",
    "Lanes",
    "Width"
]

print("1. Loading working dataset...")
working = pd.read_excel(DATA_PATH)


available_cols = [c for c in corr_columns if c in working.columns]
print(f"2. Computing correlation matrix for {len(available_cols)} variables: {', '.join(available_cols)}")
corr = working[available_cols].apply(pd.to_numeric, errors="coerce").corr()

nan_cols = corr.columns[corr.isna().all()].tolist()
if nan_cols:
    print(f"Note: Excluding variables with zero variance / undefined correlation: {nan_cols}")
    corr = corr.dropna(how="all", axis=0).dropna(how="all", axis=1)

print("\n=== Pearson Correlation Matrix ===")
print(corr.round(3).to_string())


csv_path = OUTPUT_RESULTS_DIR / "correlation_matrix.csv"
try:
    corr.round(4).to_csv(csv_path)
    print(f"\nCorrelation table saved to: {csv_path}")
except PermissionError:
    print(f"\nWarning: Could not save to {csv_path.name} because it is currently open in Excel or another program. Please close it and re-run.")


print("3. Generating correlation heatmap figure...")

fig, ax = plt.subplots(figsize=(9, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.7,
    annot_kws={"size": 10},
    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
    ax=ax
)

ax.set_title("Correlation Matrix of Traffic Variables", fontsize=14, pad=15, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()


save_path = OUTPUT_FIGURES_DIR / "correlation_matrix.png"
print("4. Saving heatmap figure (DPI=300)...")
fig.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Correlation matrix heatmap saved to: {save_path}")

plt.show()
plt.close(fig)

