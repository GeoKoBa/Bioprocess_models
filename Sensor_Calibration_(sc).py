import glob
import os
import pandas as pd
import numpy as np
import datetime
from scipy.stats import zscore
import matplotlib.pyplot as plt  # Add this import at the top of your file
from SQ_functions import (
    constrained_polynomial_fit_massbalance,
    constrained_polynomial_fit_massbalance_nonneg,
    evaluate_constrained_fit,
    export_constrained_equations,
    plot_constrained_surface,
    train_unconstrained_models,
    export_unconstrained_equations,
    plot_deviation_heatmap, 
    plot_mass_balance_violation
)

# --- CONSTANTS ---
min_degree = 1
max_degree = 4
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
z_threshold = 1.0
append = False

# --- VARIABLES & PATHS ---
gas = input("Which gas to calibrate (CO2, Air, or CO2&Air)? ").strip()
side = input("Choose inlet or outlet:").strip().lower()
sensor = input("Sensor Name (e.g., FS4003, SFC6000D or SFM3000):").strip()
base_directory = "C:/Users/GeorgiosBalamotis/sqale.ai/3-PRODUCT DEVELOPMENT - PROJECT-101-eNose - PROJECT-101-eNose/2024 12 16-Experimental data generated/Experiment_3_Partial_CO2_MF_KU/3_Sensor_Calibrations"
 
gas = gas.strip().lower()
use = f"{sensor}_{gas}_{'in' if side == 'inlet' else 'out'}"

csv_directory = os.path.join(base_directory, use)
output_directory = os.path.join(csv_directory, "Results")
os.makedirs(output_directory, exist_ok=True)

eq_file = os.path.join(output_directory, f"Equations_{current_date}.txt")

# --- LOAD ALL CSVs ---
required_cols = ["Mass_Flow_SLPM", "CO2_Conc_%", "Real_Value_CO2", "Real_Value_Air"]
csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {csv_directory}")

dfs = []

for file in csv_files:
    df_part = pd.read_csv(file, delimiter=';')
    df_part.columns = df_part.columns.str.strip()

    # Rename known alternate column names if needed
    df_part.rename(columns={
        "CO2_True": "Real_Value_CO2",
        "Air_True": "Real_Value_Air",
        "CO2_PMF": "Real_Value_CO2",
        "Air_PMF": "Real_Value_Air"
    }, inplace=True)

    dfs.append(df_part)

# Combine all into one DataFrame
df = pd.concat([df for df in dfs if not df.empty], ignore_index=True)

# Replace inf and drop rows missing required values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=required_cols)

# Confirm all required columns exist
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")
 
# --- Z-SCORE ---
cleaned_data = pd.DataFrame(columns=df.columns)
for real_val, group in df.groupby("Real_Value_CO2"):
    if len(group) > 1:
        std_dev = group["Mass_Flow_SLPM"].std()

        if std_dev < 1e-10:
            cleaned_data = pd.concat([cleaned_data, group], ignore_index=True)
            continue

        z_scores = np.abs(zscore(group[["Mass_Flow_SLPM"]]))
        group["Z_Score"] = z_scores

        removed_points = group[group["Z_Score"] > z_threshold]
        kept_points = group[group["Z_Score"] <= z_threshold]

        # Plot removed and kept points
        plt.figure(figsize=(8, 6))
        if not removed_points.empty:
            plt.scatter(
                removed_points["Mass_Flow_SLPM"],
                removed_points["CO2_Conc_%"],
                color="red",
                label="Removed Points",
                alpha=0.7,
            )
        if not kept_points.empty:
            plt.scatter(
                kept_points["Mass_Flow_SLPM"],
                kept_points["CO2_Conc_%"],
                color="green",
                label="Kept Points",
                alpha=0.7,
            )

        plt.title(f"Z-Test Results for Real_Value_CO2 = {real_val}")
        plt.xlabel("Mass_Flow_SLPM")
        plt.ylabel("CO2_Conc_%")
        plt.legend()
        plt.grid(True)
        # plot_path = os.path.join(output_directory, f"Z_Test_Real_Value_CO2_{real_val}.png")
        # plt.show()
        # plt.savefig(plot_path)
        plt.close()

        if not kept_points.empty:
            # Drop all-NaN columns
            kept_points_clean = kept_points.dropna(axis=1, how="all")
            print(f"[DEBUG] Real_Value: {real_val} | kept_points_clean shape: {kept_points_clean.shape}")
            # Ensure same structure
            kept_points_clean = kept_points_clean.reindex(columns=cleaned_data.columns)

            # Drop rows that are all NaN
            kept_points_clean = kept_points_clean.dropna(how="all")

            if not kept_points_clean.empty:
                # Only concat if both DataFrames are non-empty
                if cleaned_data.empty:
                    cleaned_data = kept_points_clean.copy()
                else:
                    cleaned_data = pd.concat([cleaned_data, kept_points_clean], ignore_index=True)

# Update the main dataframe
df = cleaned_data.copy()

# Prepare inputs based on type
if gas == "co2&air":
    X = df[["Mass_Flow_SLPM", "CO2_Conc_%"]].values
else:
    X = df[["Mass_Flow_SLPM"]]
Y_co2 = df["Real_Value_CO2"].values
Y_air = df["Real_Value_Air"].values

# --- FIT CONSTRAINED MODELS ---
X_df = df[["Mass_Flow_SLPM", "CO2_Conc_%"]]  # for plotting
X = X_df.values  # for constrained model
results = []

if gas == "co2&air":
    print("Fitting constrained models for CO2 and Air...")
    for degree in range(min_degree, max_degree + 1):
        w_co2, w_air, poly = constrained_polynomial_fit_massbalance(X, Y_co2, Y_air, degree) 
      # w_co2, w_air, poly = constrained_polynomial_fit_massbalance_nonneg(X, Y_co2, Y_air, degree)        
        metrics = evaluate_constrained_fit(X, Y_co2, Y_air, w_co2, w_air, poly)

        X_poly = poly.fit_transform(X)
        df[f"CO2_PMF_Pred_Deg{degree}"] = X_poly @ w_co2
        df[f"Air_PMF_Pred_Deg{degree}"] = X_poly @ w_air

        results.append(metrics)

        # Visualization
        plot_constrained_surface(w_co2, poly, "CO2", degree, output_directory, X_df.values, Y_co2)
        plot_constrained_surface(w_air, poly, "Air", degree, output_directory, X_df.values, Y_air)

#        plot_constrained_surface(w_co2, poly, "CO2", degree, output_directory, X_df, Y_co2)
#        plot_constrained_surface(w_air, poly, "Air", degree, output_directory, X_df, Y_air)

        # Save equations and metrics
        export_constrained_equations(w_co2, w_air, poly, eq_file,
                                     degree=degree, metrics=metrics, append=append)
        append = True
    
# --- POST-FITTING DIAGNOSTIC PLOTS ---
    df["Mass_Flow"] = df["Mass_Flow_SLPM"]
    df["CO2_Concentration"] = df["CO2_Conc_%"]

    plot_mass_balance_violation(df, min_degree, max_degree)

    # Plot deviation heatmaps (per gas, per degree)
    for degree in range(min_degree, max_degree + 1):
        for gas_name in ["CO2", "Air"]:
            col_pred = f"{gas_name}_PMF_Pred_Deg{degree}"
            if col_pred in df.columns and "Mass_Flow" in df.columns:
                df[f"{gas_name}_Deviation_Deg{degree}"] = 100 * (
                    (df[col_pred] - df[f"Real_Value_{gas_name}"]) / df[f"Real_Value_{gas_name}"]
                )
                plot_deviation_heatmap(df, f"{gas_name}_Deviation_Deg{degree}", f"{gas_name} Deviation - Degree {degree}")
        
if gas == "co2":
    print ("Fitting unconstrained model for CO2...")
    df, models, polynomials, metrics = train_unconstrained_models(
        df, feature_names=X.columns.tolist()
    )
    export_unconstrained_equations(
        models, polynomials, metrics, X.columns.tolist(), eq_file
    )
