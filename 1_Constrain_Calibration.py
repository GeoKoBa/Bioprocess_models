import pandas as pd
import numpy as np
import os
import re
import plotly.express as px
from datetime import datetime
from SQ_functions import (
    constrained_polynomial_fit_massbalance,
    constrained_polynomial_fit_massbalance_nonneg,
    evaluate_constrained_fit,
    export_constrained_equations,
    export_to_excel,
    plot_constrained_surface,
    plot_deviation_heatmap,
    get_datasets,
    stress_test_constrained,
    merge_sensor_pairs,
    plot_mass_balance_violation,
)

# --------------------------------------------------------------------------------
# USER PROMPTS
get_plots = input("Would you like to generate plots? (yes/no): ").strip().lower() == "yes"
run_stress_test = input("Would you like to run the STRESS TEST section? (yes/no): ").strip().lower() == "yes"

# DIRECTORIES
general_directory = 'C:/Users/GeorgiosBalamotis/sqale.ai/3-PRODUCT DEVELOPMENT - PROJECT-101-eNose - PROJECT-101-eNose/2024 12 16-Experimental data generated/Experiment_3_Partial_CO2_MF_KU/1_Pre_wiffs'
csv_directory = os.path.join(general_directory, 'Datasets')
output_dir = os.path.join(general_directory, 'Results', 'Constrain_incl')
os.makedirs(output_dir, exist_ok=True)

# TIME VARIABLES:
current_datetime = datetime.now()
current_date = current_datetime.strftime('%Y-%m-%d')
current_time = current_datetime.strftime('%H:%M:%S')

# CONSTANTS:
min_order = 1
max_order = 4
append = False


# --------------------------------------------------------------------------------
(sfmmf_co2_files, alicat_air_files, sfmmf_air_files, alicat_co2_files, 
 sfmmf_both_varying_files, alicat_bv_air_files, alicat_bv_co2_files,
 sfmmf_co2_count, alicat_air_count, sfmmf_air_count, alicat_co2_count, 
 sfmmf_bv_count, alicat_bv_air_count, alicat_bv_co2_count) = get_datasets(csv_directory)

all_merged_data = merge_sensor_pairs(
    sfmmf_co2_files, alicat_air_files,
    sfmmf_air_files, alicat_co2_files,
    sfmmf_both_varying_files, alicat_bv_air_files, alicat_bv_co2_files
)

# Extra columns and Percentile differences
all_merged_data['CO2_PMFout_calc'] = all_merged_data['Mass_Flow'] * all_merged_data['CO2_Concentration']
all_merged_data['Air_PMFout_calc'] = all_merged_data['Mass_Flow'] * (1 - all_merged_data['CO2_Concentration'])
all_merged_data['CO2_Calc_Diff'] = (all_merged_data['CO2_PMFout_calc'] - all_merged_data['CO2 PMF [SLPM]']) / all_merged_data['CO2 PMF [SLPM]']
all_merged_data['Air_Calc_Diff'] = (all_merged_data['Air_PMFout_calc'] - all_merged_data['Air PMF [SLPM]']) / all_merged_data['Air PMF [SLPM]']

# --------------------------------------------------------------------------------
X_data = all_merged_data[["Mass_Flow", "CO2_Concentration"]].values
Y_co2 = all_merged_data["CO2 PMF [SLPM]"].values
Y_air = all_merged_data["Air PMF [SLPM]"].values

if np.isnan(X_data).any() or np.isnan(Y_co2).any() or np.isnan(Y_air).any():
    raise ValueError("NaNs detected in input data. Please clean the dataset.")

results = []
excel_path = os.path.join(output_dir, f"Data_Analysis_{current_date}.xlsx")

w_co2_dict = {}
w_air_dict = {}
poly_dict = {}

for degree in range(min_order, max_order + 1):
    # w_co2, w_air, poly = constrained_polynomial_fit_massbalance(X_data, Y_co2, Y_air, degree)
    w_co2, w_air, poly = constrained_polynomial_fit_massbalance_nonneg(X_data, Y_co2, Y_air, degree)
    metrics = evaluate_constrained_fit(X_data, Y_co2, Y_air, w_co2, w_air, poly, degree=degree)

    all_merged_data[f"Mass_Balance_Violation_Deg{degree}"] = metrics["Constraint_Violation"]

    w_co2_dict[degree] = w_co2
    w_air_dict[degree] = w_air
    poly_dict[degree] = poly

    mask = (~np.isnan(X_data).any(axis=1)) & (~np.isinf(X_data).any(axis=1)) &            (~np.isnan(Y_co2)) & (~np.isinf(Y_co2)) &            (~np.isnan(Y_air)) & (~np.isinf(Y_air))

    all_merged_data[f"CO2_PMF_Pred_Deg{degree}"] = np.nan
    all_merged_data[f"Air_PMF_Pred_Deg{degree}"] = np.nan
    all_merged_data[f"CO2_Pred_Diff_Deg{degree}"] = np.nan
    all_merged_data[f"Air_Pred_Diff_Deg{degree}"] = np.nan

    all_merged_data.loc[mask, f"CO2_PMF_Pred_Deg{degree}"] = metrics["CO2_Pred"]
    all_merged_data.loc[mask, f"Air_PMF_Pred_Deg{degree}"] = metrics["Air_Pred"]
    all_merged_data.loc[mask, f"CO2_Pred_Diff_Deg{degree}"] = (metrics["CO2_Pred"] - Y_co2[mask]) / Y_co2[mask]
    all_merged_data.loc[mask, f"Air_Pred_Diff_Deg{degree}"] = (metrics["Air_Pred"] - Y_air[mask]) / Y_air[mask]

    eq_path = os.path.join(output_dir, f"Polynomial_Equations_Constrained_{current_date}.txt")
    export_constrained_equations(
        w_co2, w_air, poly,
        output_path=eq_path,
        degree=degree,
        metrics=metrics,
        append=append
    )
    append = True 

    if get_plots:
        plot_constrained_surface(w_co2, poly, "CO2", degree, output_dir, X_data, Y_co2)
        plot_constrained_surface(w_air, poly, "Air", degree, output_dir, X_data, Y_air)
        plot_deviation_heatmap(all_merged_data, f"CO2_Pred_Diff_Deg{degree}", f"CO2 Deviation Deg {degree}")
        plot_deviation_heatmap(all_merged_data, f"Air_Pred_Diff_Deg{degree}", f"Air Deviation Deg {degree}")

    results.append(metrics)

if get_plots:
    plot_mass_balance_violation(all_merged_data, min_order, max_order)

# --------------------------------------------------------------------------------
# Column reordering and export
columns_order = [
    'Mass_Flow', 'CO2_Concentration',
    'CO2 PMF [SLPM]', 'Air PMF [SLPM]',
    'CO2_PMFout_calc', 'Air_PMFout_calc',
    'CO2_Calc_Diff', 'Air_Calc_Diff'
]
for degree in range(min_order, max_order + 1):
    columns_order.extend([
        f"CO2_PMF_Pred_Deg{degree}",
        f"Air_PMF_Pred_Deg{degree}",
        f"CO2_Pred_Diff_Deg{degree}",
        f"Air_Pred_Diff_Deg{degree}"
    ])
all_merged_data = all_merged_data[columns_order]
all_merged_data.sort_values(by='Mass_Flow', inplace=True)

# Export
export_to_excel(excel_path, {"Data_analysis": all_merged_data})
       
# STRESS TEST PART
if run_stress_test:
    stress_test_constrained(w_co2_dict, w_air_dict, poly_dict, output_dir, min_order, max_order, current_date)
else:
    print("Skipping Stress Test!")