# indian-toll-plaza-analytics
# Toll Traffic Research Project

A transportation engineering research project focused on analyzing highway traffic characteristics, peak-hour flow dynamics, and vehicle equivalence modeling using real-world traffic data collected from Indian National Highway toll plazas.

---

## About the Project

This project studies vehicular traffic flow patterns across Indian National Highways to understand how traffic volume and vehicle composition affect highway capacity and toll operations. 

Key objectives include:
- **Peak-Hour Factor ($K$-Factor) Modeling**: Analyzing how the proportion of daily traffic occurring during peak hours varies with total traffic volume and commercial freight share.
- **Passenger Car Unit (PCU) Equivalence**: Converting heterogeneous vehicle streams (cars, buses, trucks, multi-axle vehicles) into standard PCU values following Indian Highway Capacity Manual (Indo-HCM) standards.
- **Predictive Model Development**: Building and comparing parametric regression models and machine learning algorithms (Linear Regression, Power Regression, Random Forest, XGBoost, and Neural Networks) to estimate $K$-factor and PCU.

---

## Dataset

The analysis is based on continuous 7-day (168 hours per location) traffic survey data recorded using Automatic Traffic Counter and Classifier (ATCC) systems across 8 National Highway toll plazas in India:

- **NH-75** (Karnataka - 2 locations)
- **NH-16** (Ranasthalam, Andhra Pradesh)
- **NH-65** (Chintapally, Telangana)
- **NH-163 / NH-44** (Muthojipet, Telangana)
- **NH-44** (Seoni, Madhya Pradesh)
- **State Arterials** (Chintalapalem & Rapur, Andhra Pradesh)

The survey classifies traffic into 8 vehicle categories: Cars, Buses, Light Commercial Vehicles (LCV), 2-Axle Trucks, 3-Axle Trucks, Multi-Axle Vehicles (MAV), Tractors, and Others.

---

## Project Structure

```text
├── 01_Original_Location_Files/   # Raw 7-day ATCC Excel survey files
├── 02_Master_Database/           # Compiled master database of traffic variables
├── 03_Data_Dictionary/           # Variable definitions and units
├── 04_Working_Dataset/           # Processed working dataset script and Excel file
├── 05_Analysis_Code/             # Statistical analysis (correlation, scatter plots, curve fitting)
├── 06_Results/                   # Summary tables and figures
├── 07_Model_Development/         # Model implementations (MLR, Power Regression, RF, XGBoost, MLP)
├── 08_Model_Evaluation/          # Cross-validation and evaluation pipeline
├── requirements.txt              # Python package dependencies
└── README.md                     # Project documentation

