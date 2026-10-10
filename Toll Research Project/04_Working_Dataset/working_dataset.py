from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_DIR = BASE_DIR / "02_Master_Database"
OUTPUT = BASE_DIR / "04_Working_Dataset" / "working_dataset.xlsx"

files = [
    f for ext in ("*.xlsx", "*.xls")
    for f in MASTER_DIR.glob(ext)
    if not f.name.startswith("~$")
]

if not files:
    raise FileNotFoundError(f"No Excel file found in {MASTER_DIR}")

source = files[0]

try:
    xls = pd.ExcelFile(source)
    sheet = (
        "Traffic_Master_Database"
        if "Traffic_Master_Database" in xls.sheet_names
        else 0
    )
    master = pd.read_excel(source, sheet_name=sheet)
    xls.close()

except PermissionError:
    raise PermissionError(
        f"\nCannot access:\n{source}\n\n"
        "Close the Excel file and make sure OneDrive has finished syncing."
    )

mapping = {
    "Number of Lanes": "Lanes",
    "Road Width (m)": "Width",
    "Speed Limit (km/h)": "Speed Limit",
    "AADT/ADT": "ADT",
    "Weekly Traffic": "Weekly Traffic",
    "Peak Hour Volume": "PHV",
    "Peak Hour Factor (PHF)": "PHF",
    "K-Factor": "K",
    "D-Factor": "D",
    "PCU": "PCU",
    "Heavy Vehicle (%)": "HV%",
    "Cars": "Cars",
    "Bus": "Bus",
    "LCV": "LCV",
    "MAV": "MAV",
    "2-Axle Truck": "2-Axle Truck",
    "3-Axle Truck": "3-Axle Truck",
    "4-6 Axle Truck": "4-6 Axle Truck",
    "7 Axle Truck": "7 Axle Truck"
}

cols = {k: v for k, v in mapping.items() if k in master.columns}
working = master[list(cols)].rename(columns=cols)

for col in working.columns:
    working[col] = pd.to_numeric(working[col], errors="coerce")

for col in ["Cars", "Bus", "LCV", "MAV"]:
    if col in working:
        working[f"{col}%"] = (working[col] / working["ADT"] * 100).round(3)

trucks = [
    c for c in ["2-Axle Truck", "3-Axle Truck",
                "4-6 Axle Truck", "7 Axle Truck"]
    if c in working
]

if trucks:
    working["Truck"] = working[trucks].sum(axis=1)
    working["Truck%"] = (working["Truck"] / working["ADT"] * 100).round(3)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
working.to_excel(OUTPUT, index=False)

print(f"Created: {OUTPUT}")
print(f"Shape: {working.shape}")