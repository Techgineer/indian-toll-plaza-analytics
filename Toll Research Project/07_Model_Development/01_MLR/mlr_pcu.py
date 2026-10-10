from pathlib import Path
import pandas as pd
import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR.parent.parent / "04_Working_Dataset" / "working_dataset.xlsx"

print("Reading File From :")
print(file_path)

print("File exists:",file_path.exists())

working = pd.read_excel(file_path)


X = working[["ADT", "HV%"]]


X = sm.add_constant(X)

y = working["PCU"]


model = sm.OLS(y, X).fit()

print("==== MLR MODEL : PCU ====")
print(model.summary())


working["PCU_Predicted"] = model.predict(X)

output_file = BASE_DIR / "mlr_model_pcu.xlsx"
working.to_excel(output_file, index=False)

print("\nOutput saved to:")
print(output_file)