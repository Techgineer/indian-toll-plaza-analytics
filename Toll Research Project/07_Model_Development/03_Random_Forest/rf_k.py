from pathlib import Path
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent 

file_path = (BASE_DIR.parent.parent / "04_Working_Dataset" / "working_dataset.xlsx")

working = pd.read_excel(file_path)

X = working[["ADT","HV%"]]

y = working["K"]

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    RandomForestRegressor(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=1,
        random_state=42,
    )
)

model.fit(X, y)

working["K_Predicted_RF"] = model.predict(X)

result_dir = BASE_DIR / "results"
result_dir.mkdir(exist_ok=True)

output_file = result_dir / "rf_k_results.xlsx"
working.to_excel(output_file,index=False)

print("Random Forest Model Completed")
print(f"Results saved to: {output_file}")