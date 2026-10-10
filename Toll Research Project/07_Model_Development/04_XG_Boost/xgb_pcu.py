from pathlib import Path
import pandas as pd 
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent

file_path = (BASE_DIR.parent.parent / "04_Working_Dataset" / "working_dataset.xlsx")

working = pd.read_excel(file_path)

X = working[["ADT","HV%"]]
y = working["PCU"]

import numpy as np
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(
    regressor=xgb.XGBRegressor(
        n_estimators=150,
        max_depth=2,
        learning_rate=0.08,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror"
    ),
    func=np.log,
    inverse_func=np.exp
)

model.fit(X, y)

working["PCU_Predicted_XGB"] = model.predict(X)

result_dir = BASE_DIR / "results"
result_dir.mkdir(exist_ok=True)

output_file = result_dir / "xgb_pcu_results.xlsx"
working.to_excel(output_file,index=False)

print("XGBoost PCU Model Completed")
print(f"Results saved to: {output_file}")
