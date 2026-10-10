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
y = working["K"]

from sklearn.preprocessing import PolynomialFeatures

X_log = np.log(X)
y_log = np.log(y)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_log_poly = poly.fit_transform(X_log)
feature_names = poly.get_feature_names_out(["ln_ADT", "ln_HV%"])
X_log_poly_df = pd.DataFrame(X_log_poly, columns=feature_names)

X_log_const = sm.add_constant(X_log_poly_df)

model = sm.OLS(y_log, X_log_const).fit()

print("=== TRANSLOG POWER REGRESSION MODEL : K ===")
print(model.summary())

log_predicted = model.predict(X_log_const)

working["K_predicted"] = np.exp(log_predicted)

working["K_residual"] = working["K"] - working["K_predicted"]

results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)

output_file = results_dir / "pr_k_results.xlsx"

working.to_excel(output_file, index=False)

print("\nResults saved to:")
print(output_file)

