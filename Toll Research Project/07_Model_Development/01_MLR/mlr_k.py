from pathlib import Path
import pandas as pd
import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR.parent.parent / "04_Working_Dataset" / "working_dataset.xlsx"

print("Reading File From :")
print(file_path)

print("File exists:",file_path.exists())

working = pd.read_excel(file_path)


from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(working[["ADT", "HV%"]])
feature_names = poly.get_feature_names_out(["ADT", "HV%"])
X_poly_df = pd.DataFrame(X_poly, columns=feature_names)

X_const = sm.add_constant(X_poly_df)
y = working["K"]

model = sm.OLS(y, X_const).fit()

print("==== MLR MODEL (POLYNOMIAL) : K ====")
print(model.summary())

working["K_Predicted"] = model.predict(X_const)

output_file = BASE_DIR / "mlr_model_k.xlsx"
working.to_excel(output_file, index=False)

print("\nOutput saved to:")
print(output_file)
