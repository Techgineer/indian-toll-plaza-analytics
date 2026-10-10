import numpy as np
import pandas as pd 
import statsmodels.api as sm
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR.parent.parent / "04_Working_Dataset" / "working_dataset.xlsx"

print("Reading File From :")
print(file_path)
print("File exists:",file_path.exists())

working = pd.read_excel(file_path)


X = working[["ADT","HV%"]].copy()
y = working["PCU"]

X_log = np.log(X)
y_log = np.log(y)

X_log = sm.add_constant(X_log)

model = sm.OLS(y_log, X_log).fit()

print("=== POWER REGRESSION MODEL : PCU ===")
print(model.summary())

log_predicted = model.predict(X_log)

working["PCU_predicted"] = np.exp(log_predicted)

working["PCU_residual"] = working["PCU"] - working["PCU_predicted"]

results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)

output_file = results_dir / "pr_pcu_results.xlsx"

working.to_excel(output_file, index=False)

print("\nResults saved to:")
print(output_file)