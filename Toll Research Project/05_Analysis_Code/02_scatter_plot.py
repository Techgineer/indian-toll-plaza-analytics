from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "04_Working_Dataset" / "working_dataset.xlsx"
OUTPUT_DIR = BASE_DIR / "06_Results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


working = pd.read_excel(DATA_PATH)


relationships = [
    ("HV%", "K"),
    ("HV%", "PHF"),
    ("HV%", "PCU"),
    ("ADT", "K"),
    ("ADT", "PHV"),
    ("ADT", "PCU")
]

sns.set_theme(style="whitegrid")

for x, y in relationships:
    if x not in working.columns or y not in working.columns:
        print(f"Skipping ({x}, {y}): one or both columns not found in dataset.")
        continue

    data = working[[x, y]].dropna()
    if data.empty:
        print(f"Skipping ({x}, {y}): no valid data points.")
        continue

    fig, ax = plt.subplots(figsize=(8, 6))

    
    sns.regplot(
        data=data,
        x=x,
        y=y,
        scatter_kws={"alpha": 0.7, "color": "#1f77b4"},
        line_kws={"color": "#d62728", "linewidth": 2},
        ax=ax
    )

    ax.set_title(f"Scatter Plot of {y} vs {x}", fontsize=13, pad=10)
    ax.set_xlabel(x, fontsize=11)
    ax.set_ylabel(y, fontsize=11)
    plt.tight_layout()
    clean_x = x.replace("%", "pct")
    clean_y = y.replace("%", "pct")
    save_path = OUTPUT_DIR / f"scatter_{clean_x}_{clean_y}.png"

    
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path.name}")

    plt.show()
    plt.close(fig)