from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler



torch.manual_seed(42)
np.random.seed(42)



BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

file_path = PROJECT_ROOT / "04_Working_Dataset" / "working_dataset.xlsx"

print("BASE_DIR:", BASE_DIR)
print("FILE:", file_path)
print("EXISTS:", file_path.exists())

working = pd.read_excel(file_path)

print("Dataset loaded successfully")
print(f"Number of rows: {len(working)}")
print(f"Number of columns: {len(working.columns)}")



X = working[["ADT", "HV%"]].values
y = working["PCU"].values

# Standardize inputs and target
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)


class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, X):
        return self.network(X)


model = MLP()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

epochs = 600

model.train()

for epoch in range(epochs):

    predictions = model(X_tensor)

    loss = criterion(predictions, y_tensor)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss.item():.6f}"
        )


model.eval()

with torch.no_grad():
    predictions_scaled = model(X_tensor).numpy()

# Inverse transform back to original PCU scale
predicted_pcu = scaler_y.inverse_transform(predictions_scaled).flatten()

working["PCU_Predicted_MLP"] = predicted_pcu
working["PCU_Residual_MLP"] = working["PCU"] - working["PCU_Predicted_MLP"]

# Performance Metrics
mae = np.mean(np.abs(working["PCU_Residual_MLP"]))
rmse = np.sqrt(np.mean(working["PCU_Residual_MLP"] ** 2))
ss_tot = np.sum((working["PCU"] - np.mean(working["PCU"])) ** 2)
ss_res = np.sum(working["PCU_Residual_MLP"] ** 2)
r2 = 1 - (ss_res / ss_tot)

print("\n==== MLP MODEL PERFORMANCE (PCU) ====")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")

result_dir = BASE_DIR / "results"
result_dir.mkdir(exist_ok=True)

output_file = result_dir / "mlp_pcu_results.xlsx"

try:
    working.to_excel(output_file, index=False)
    print(f"\nMLP Model Completed")
    print(f"Results saved to: {output_file}")
except PermissionError:
    alt_file = result_dir / "mlp_pcu_results_updated.xlsx"
    working.to_excel(alt_file, index=False)
    print(f"\n[Warning] '{output_file.name}' is open in Excel.")
    print(f"Saved to alternate file instead: {alt_file}")