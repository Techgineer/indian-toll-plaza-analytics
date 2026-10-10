from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "04_Working_Dataset" / "working_dataset.xlsx"
OUTPUT_FIGURES_DIR = BASE_DIR / "06_Results" / "figures"
OUTPUT_DIR = BASE_DIR / "06_Results"
OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


working = pd.read_excel(DATA_PATH)


def linear(x, a, b):
    return a + b * x

def power(x, a, b):
    return a * (x ** b)

def exponential(x, a, b):
    return a * np.exp(b * x)


x_raw = pd.to_numeric(working["ADT"], errors="coerce").to_numpy()
y_raw = pd.to_numeric(working["K"], errors="coerce").to_numpy()


valid_base = np.isfinite(x_raw) & np.isfinite(y_raw)
x_clean = x_raw[valid_base]
y_clean = y_raw[valid_base]

models = [
    {
        "name": "Linear",
        "func": linear,
        "p0": [np.mean(y_clean), 0.0],
        "strictly_positive": False,
        "format": lambda p: f"K = {p[0]:.4f} + ({p[1]:.2e}) * ADT"
    },
    {
        "name": "Power",
        "func": power,
        "p0": [0.1, -0.05],
        "strictly_positive": True,
        "format": lambda p: f"K = {p[0]:.4f} * ADT^({p[1]:.4f})"
    },
    {
        "name": "Exponential",
        "func": exponential,
        "p0": [np.mean(y_clean), 0.0],
        "strictly_positive": True,
        "format": lambda p: f"K = {p[0]:.4f} * exp({p[1]:.2e} * ADT)"
    }
]

results = []
fig, ax = plt.subplots(figsize=(10, 6))


ax.scatter(x_clean, y_clean, color="#1f77b4", alpha=0.7, edgecolors="k", s=45, label="Observed Data", zorder=3)


x_line = np.linspace(x_clean.min(), x_clean.max(), 400)

for m in models:
    name = m["name"]
    func = m["func"]
    p0 = m["p0"]
    fmt = m["format"]

    
    if m["strictly_positive"]:
        mask = (x_clean > 0) & (y_clean > 0)
    else:
        mask = np.ones(len(x_clean), dtype=bool)

    x_fit = x_clean[mask]
    y_fit = y_clean[mask]

    try:
        params, pcov = curve_fit(func, x_fit, y_fit, p0=p0, maxfev=20000)
        y_pred = func(x_fit, *params)

        
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        rmse = np.sqrt(np.mean((y_fit - y_pred) ** 2))

        eq_str = fmt(params)
        results.append({
            "Model": name,
            "Equation": eq_str,
            "R²": round(r2, 4),
            "RMSE": round(rmse, 4),
            "Params": [round(float(p), 6) for p in params]
        })

        
        ax.plot(x_line, func(x_line, *params), label=f"{name} (R² = {r2:.3f})", linewidth=2.2)

    except Exception as e:
        print(f"Fit failed for {name}: {e}")


results_df = pd.DataFrame(results)
print("\n=== Model Fitting Summary: K-Factor vs ADT ===")
print(results_df[["Model", "Equation", "R²", "RMSE"]].to_string(index=False))


summary_csv_path = OUTPUT_DIR / "equation_fitting_summary.csv"
results_df.to_csv(summary_csv_path, index=False)
print(f"\nFitted parameters and metrics saved to: {summary_csv_path}")


ax.set_title("Fitted Equations: K-Factor vs Average Daily Traffic (ADT)", fontsize=13, pad=12)
ax.set_xlabel("Average Daily Traffic (ADT, vehicles/day)", fontsize=11)
ax.set_ylabel("K-Factor", fontsize=11)
ax.legend(frameon=True)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

figure_path = OUTPUT_FIGURES_DIR / "equation_fitting_ADT_K.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight")
print(f"Fitted curve plot saved to: {figure_path}")

plt.show()
plt.close(fig)