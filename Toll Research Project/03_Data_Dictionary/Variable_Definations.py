import pandas as pd


file_path = "02_Master_Database/Traffic_Master_Database (1).xlsx"


master = pd.read_excel(file_path, sheet_name="Traffic_Master_Database")


dictionary = pd.DataFrame({
    "Variable": master.columns,
    "Data Type": master.dtypes.astype(str).values,
    "Unit": "",
    "Definition": "", 
})

definitions = {
    "Location ID": ("—", "Unique identifier assigned to each location"),
    "Location Name": ("—", "Name of the traffic survey location"),
    "Highway Number": ("—", "Identification number of the highway"),
    "State": ("—", "State where the location is situated"),
    "Latitude": ("Decimal degrees", "Geographic latitude of the location"),
    "Longitude": ("Decimal degrees", "Geographic longitude of the location"),
    "Area Type (Urban/Rural)": ("—", "Classification of the surrounding area as urban or rural"),
    "Facility Type (Toll/Midblock/Intersection)": ("—", "Type of traffic facility being studied"),
    "Number of Lanes": ("lanes", "Number of traffic lanes"),
    "Road Type (Divided/Undivided)": ("—", "Indicates whether the roadway is divided or undivided"),
    "Median Type": ("—", "Type of median separating opposing traffic"),
    "Road Width (m)": ("m", "Width of the roadway"),
    "Speed Limit (km/h)": ("km/h", "Posted speed limit"),
    "Observation Start Date": ("date", "Date on which traffic observation began"),
    "Observation End Date": ("date", "Date on which traffic observation ended"),
    "Observation Duration (Days)": ("days", "Total duration of the observation period"),
    "Hourly Traffic (Avg)": ("vehicles/hour", "Average hourly traffic volume"),
    "Daily Traffic": ("vehicles/day", "Total traffic volume during one day"),
    "Weekly Traffic": ("vehicles/week", "Total traffic volume during one week"),
    "AADT/ADT": ("vehicles/day", "Average annual daily traffic or average daily traffic"),
    "Direction 1 Traffic": ("vehicles/day", "Traffic volume traveling in direction 1"),
    "Direction 2 Traffic": ("vehicles/day", "Traffic volume traveling in direction 2"),
    "Directional Split (%)": ("%", "Percentage distribution of traffic between directions"),
    "Cars": ("vehicles/day", "Number of cars observed"),
    "Two Wheelers": ("vehicles/day", "Number of two-wheelers observed"),
    "Auto Rickshaws": ("vehicles/day", "Number of auto rickshaws observed"),
    "LCV": ("vehicles/day", "Number of light commercial vehicles observed"),
    "Bus": ("vehicles/day", "Number of buses observed"),
    "2-Axle Truck": ("vehicles/day", "Number of trucks with two axles observed"),
    "3-Axle Truck": ("vehicles/day", "Number of trucks with three axles observed"),
    "MAV": ("vehicles/day", "Number of multi-axle vehicles observed"),
    "Tractor": ("vehicles/day", "Number of tractors observed"),
    "Others": ("vehicles/day", "Number of vehicles not included in the listed categories"),
    "Total Commercial Vehicles": ("vehicles/day", "Total number of commercial vehicles"),
    "Heavy Vehicle (%)": ("%", "Percentage of heavy vehicles in total traffic"),
    "PCU": ("PCU/day", "Traffic volume converted into passenger car units"),
    "Peak Hour": ("time", "Time period with the highest traffic volume"),
    "Peak Hour Volume": ("vehicles/hour", "Traffic volume during the peak hour"),
    "Peak Hour Factor (PHF)": ("ratio", "Measure of traffic variation within the peak hour"),
    "K-Factor": ("ratio", "Ratio of peak-hour traffic to daily traffic"),
    "D-Factor": ("ratio", "Proportion of traffic traveling in the major direction"),
    "Weekend Avg Traffic": ("vehicles/day", "Average traffic volume on weekend days"),
    "Weekday Avg Traffic": ("vehicles/day", "Average traffic volume on weekday days"),
    "Max Hourly Volume": ("vehicles/hour", "Highest observed hourly traffic volume"),
    "Min Hourly Volume": ("vehicles/hour", "Lowest observed hourly traffic volume"),
    "Remarks": ("—", "Additional notes or observations about the data")
}

dictionary["Unit"] = dictionary["Variable"].map(
    lambda x: definitions.get(x, ("—", ""))[0]
)

dictionary["Definition"] = dictionary["Variable"].map(
    lambda x: definitions.get(x, ("—", ""))[1]
)

dictionary.to_excel("02_Master_Database/Variable_Definations_Cleaned.xlsx", index=False)
print("File saved successfully to: 02_Master_Database/Variable_Definations_Cleaned.xlsx")


