import matplotlib.pyplot as plt
import pandas as pd
import os
from SQ_functions import assign_real_values

# --- USER INPUT ---
Ndata = input("Choose the dataset:")
date = "2025-04-03"
co2_threshold = 3.00                                            # Because the value is in %
flow_threshold = 0.01                                           # SLPM

# --- INPUT SELECTION ---
gas = input("Which gas to calibrate (CO2, Air, or CO2&Air)? ").strip()
side = input("Choose inlet or outlet: ").strip().lower()
sensor = input("Sensor Name (e.g., FS4003, SFC6000D or SFM3000): ").strip()

# --- NORMALIZATION ---
gas = gas.strip().lower()                                       # Normalize: "CO2", "Air", "CO2&Air"

# --- PATH SETUP ---
base_directory = "C:/Users/GeorgiosBalamotis/sqale.ai/3-PRODUCT DEVELOPMENT - PROJECT-101-eNose - PROJECT-101-eNose/2024 12 16-Experimental data generated/Experiment_3_Partial_CO2_MF_KU/3_Sensor_Calibrations"

use = f"{sensor}_{gas}_{'in' if side == 'inlet' else 'out'}"
file_name = f"Calibration_{gas}_{Ndata}.csv"

csv_directory = os.path.join(base_directory, use)
file_path = os.path.join(csv_directory, file_name)
output_path = file_path  # saving in same location

# Loading and treating the CSV data
df = pd.read_csv(file_path, delimiter=";")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df.columns = df.columns.str.strip()                     # Clean up column names (remove whitespace)

# Normalize column names for compatibility
column_renames = {
    "CO2_Concentration": "CO2_Conc_%",
    "Mass Flow (SFC6000D - 5 SLPM)": "Mass_Flow_SLPM",
    "MEMS_Mass_Flow_intake": "Mass_Flow_SLPM",
    "Sensirion_Mass_Flow_intake": "Mass_Flow_SLPM",
    "FS4003_MF_intake": "Mass_Flow_SLPM",
    "SFM3000_MF_intake": "Mass_Flow_SLPM"
}
df.rename(columns={col: new for col, new in column_renames.items() if col in df.columns}, inplace=True)

# --- REAL VALUES ---
new_co2_values = [
    0.0250, 0.0375, 0.0500, 0.0625, 0.0750, 0.0875, 0.1000
]

Real_Values_CO2 = [
    (f"{date} 15:14:56", new_co2_values[0]),
    (f"{date} 15:31:08", new_co2_values[1]),
    (f"{date} 15:50:36", new_co2_values[2]),
    (f"{date} 16:11:02", new_co2_values[3]),
    (f"{date} 16:32:19", new_co2_values[4]),
    (f"{date} 16:52:05", new_co2_values[5]),
    (f"{date} 17:11:01", new_co2_values[6]),
]

new_air_values = [
    0.10, 0.14, 0.18, 0.23, 0.31, 0.48, 0.15, 0.21, 0.26, 0.34, 0.46, 0.71, 0.20,
    0.28, 0.35, 0.45, 0.62, 0.95, 0.25, 0.35, 0.44, 0.56, 0.77, 1.19, 0.30, 0.43,
    0.53, 0.68, 0.93, 1.43, 0.35, 0.50, 0.61, 0.79, 1.08, 1.66, 0.40, 0.57, 0.70,
    0.90, 1.23, 1.90
]

Real_Values_Air = [
    (f"{date} 15:14:56", new_air_values[0]),
    (f"{date} 15:17:46", new_air_values[1]),
    (f"{date} 15:20:06", new_air_values[2]),
    (f"{date} 15:23:36", new_air_values[3]),
    (f"{date} 15:26:07", new_air_values[4]),
    (f"{date} 15:28:17", new_air_values[5]),
    (f"{date} 15:31:08", new_air_values[6]),
    (f"{date} 15:36:04", new_air_values[7]),
    (f"{date} 15:38:44", new_air_values[8]),
    (f"{date} 15:40:54", new_air_values[9]),
    (f"{date} 15:47:15", new_air_values[10]),
    (f"{date} 15:50:36", new_air_values[11]),
    (f"{date} 15:54:58", new_air_values[12]),
    (f"{date} 15:59:29", new_air_values[13]),
    (f"{date} 16:02:20", new_air_values[14]),
    (f"{date} 16:05:11", new_air_values[15]),
    (f"{date} 16:07:41", new_air_values[16]),
    (f"{date} 16:11:02", new_air_values[17]),
    (f"{date} 16:15:34", new_air_values[18]),
    (f"{date} 16:18:15", new_air_values[19]),
    (f"{date} 16:21:16", new_air_values[20]),
    (f"{date} 16:24:57", new_air_values[21]),
    (f"{date} 16:28:38", new_air_values[22]),
    (f"{date} 16:32:19", new_air_values[23]),
    (f"{date} 16:36:20", new_air_values[24]),
    (f"{date} 16:40:31", new_air_values[25]),
    (f"{date} 16:43:02", new_air_values[26]),
    (f"{date} 16:45:33", new_air_values[27]),
    (f"{date} 16:48:34", new_air_values[28]),
    (f"{date} 16:52:05", new_air_values[29]),
    (f"{date} 16:55:56", new_air_values[30]),
    (f"{date} 16:58:37", new_air_values[31]),
    (f"{date} 17:01:38", new_air_values[32]),
    (f"{date} 17:04:29", new_air_values[33]),
    (f"{date} 17:07:09", new_air_values[34]),
    (f"{date} 17:11:01", new_air_values[35]),
    (f"{date} 17:15:02", new_air_values[36]),
    (f"{date} 17:18:13", new_air_values[37]),
    (f"{date} 17:21:04", new_air_values[38]),
    (f"{date} 17:24:25", new_air_values[39]),
    (f"{date} 17:27:05", new_air_values[40]),
]

# Apply on the dataframe
assign_real_values(df, Real_Values_CO2, "Real_Value_CO2")

if gas == "co2&air":
    assign_real_values(df, Real_Values_Air, "Real_Value_Air")

# Cleaning rows 
df = df[
    (df["CO2_Conc_%"] >= co2_threshold) &
    (df["Mass_Flow_SLPM"] >= flow_threshold)
]                                                       # Filtering dataset

df.to_csv(output_path, index=False, sep=";")
print(df)

# --- Plotting data --- 
df.set_index("Timestamp", inplace=True)
# Create figure and axis
fig, ax1 = plt.subplots(figsize=(12, 6))
# Left Y-axis (SLPM values)
ax1.plot(df.index, df["Mass_Flow_SLPM"], label="Mass Flow (SLPM)", color="green")
ax1.plot(df.index, df["Real_Value_CO2"], label="Real CO₂ Value (SLPM)", color="black")
ax1.plot(df.index, df["Real_Value_Air"], label="Real Air Value (SLPM)", color="blue")
ax1.set_ylabel("Flow / Real Values (SLPM)")
ax1.set_xlabel("Timestamp")
ax1.grid(True)
# Right Y-axis (CO₂ %)
ax2 = ax1.twinx()
ax2.plot(df.index, df["CO2_Conc_%"], label="CO₂ Concentration (%)", color="dimgray", linestyle="--")
ax2.set_ylabel("CO₂ Concentration (%)")
# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
# Title and layout
plt.title("Mass Flow, CO₂ Concentration, and Real Values Over Time")
plt.tight_layout()
plt.show()
