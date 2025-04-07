import pandas as pd
import numpy as np
import plotly.graph_objs as go
import glob
import re
import os
import joblib
import plotly.express as px
from openpyxl import load_workbook
from openpyxl.styles import Font
from openpyxl.formatting.rule import CellIsRule
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from SQ_functions import (
    export_to_excel,
    stress_test,
    train_unconstrained_models,
    plot_unconstrained_surfaces, get_datasets, plot_deviation_heatmap,
    merge_sensor_pairs,
    export_unconstrained_equations,
    plot_mass_balance_violation
)
#--------------------------------------------------------------------------------

# Get user input to run STRESS TEST and BEST POLY sections
get_plots = input("Would you like to generate plots? (yes/no): ").strip().lower() == "yes"
run_stress_test = input("Would you like to run the STRESS TEST section? (yes/no): ").strip().lower() == "yes"

# DIRECTORIES
general_directory = 'C:/Users/GeorgiosBalamotis/sqale.ai/3-PRODUCT DEVELOPMENT - PROJECT-101-eNose - PROJECT-101-eNose/2024 12 16-Experimental data generated/Experiment_3_Partial_CO2_MF_KU'
experiment = os.path.join(general_directory, '1_Pre_wiffs')
csv_directory = os.path.join(experiment, 'Datasets')
output_dir = os.path.join(experiment, 'Results')
os.makedirs(output_dir, exist_ok=True)

# TIME VARIABLES:
current_datetime = datetime.now()
current_date = current_datetime.strftime('%Y-%m-%d')
current_time = current_datetime.strftime('%H:%M:%S')

# CONSTANTS: 
min_order = 1
max_order = 4

# --------------------------------------------------------------------------------

(sfmmf_co2_files, alicat_air_files, sfmmf_air_files, alicat_co2_files, 
 sfmmf_both_varying_files, alicat_bv_air_files, alicat_bv_co2_files,
 sfmmf_co2_count, alicat_air_count, sfmmf_air_count, alicat_co2_count, 
 sfmmf_bv_count, alicat_bv_air_count, alicat_bv_co2_count) = get_datasets(csv_directory)

# Initialize an empty DataFrame to store all merged data
all_merged_data = pd.DataFrame()
all_merged_data = merge_sensor_pairs(sfmmf_co2_files, alicat_air_files, sfmmf_air_files, alicat_co2_files, sfmmf_both_varying_files, alicat_bv_air_files, alicat_bv_co2_files)

# Extra columns and Percentile differences
all_merged_data['CO2_PMFout_calc'] = all_merged_data['Mass_Flow'] * all_merged_data['CO2_Concentration']
all_merged_data['Air_PMFout_calc'] = all_merged_data['Mass_Flow'] * (1 - all_merged_data['CO2_Concentration'])
all_merged_data['CO2_Calc_Diff'] = (all_merged_data['CO2_PMFout_calc'] - all_merged_data['CO2 PMF [SLPM]']) / all_merged_data['CO2 PMF [SLPM]']
all_merged_data['Air_Calc_Diff'] = (all_merged_data['Air_PMFout_calc'] - all_merged_data['Air PMF [SLPM]']) / all_merged_data['Air PMF [SLPM]']

# Count of NaN values and print if not zero
nan_counts = all_merged_data.isnull().sum()
if nan_counts.sum() > 0:
    print("NaN values in each column:")
    print(nan_counts)

# Align feature names with equation notation --> X is Mass_Flow, Y is CO2_Concentration
feature_names = ["Mass_Flow", "CO2_Concentration"]

# Dictionaries to store models, polynomials, and performance metrics
models = {"CO2": {}, "Air": {}}
polynomials = {"CO2": {}, "Air": {}}
metrics = {"CO2": {}, "Air": {}}  

# Prepare input and output variables for regression
X_co2 = all_merged_data[["Mass_Flow", "CO2_Concentration"]].values
Y_co2 = all_merged_data["CO2 PMF [SLPM]"].values
X_air = all_merged_data[["Mass_Flow", "CO2_Concentration"]].values
Y_air = all_merged_data["Air PMF [SLPM]"].values
columns_order = [
    'Mass_Flow', 'CO2_Concentration',
    'CO2 PMF [SLPM]', 'Air PMF [SLPM]',
    'CO2_PMFout_calc', 'Air_PMFout_calc',
    'CO2_Calc_Diff', 'Air_Calc_Diff',
]

# TRAINING NO CONSTRAINT
all_merged_data, models, polynomials, metrics = train_unconstrained_models(
    all_merged_data,
    feature_names=["Mass_Flow", "CO2_Concentration"],
    min_order=min_order,
    max_order=max_order
)

# Build the export column order dynamically
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

 # Create a grid for Total Mass Flow and CO2 Concentration

if get_plots:
    plot_unconstrained_surfaces(all_merged_data, models, polynomials, min_order, max_order)
    for degree in range(min_order, max_order + 1):
        plot_deviation_heatmap(all_merged_data, f"CO2_Pred_Diff_Deg{degree}", f"CO2_Deviation_Deg{degree}")
        plot_deviation_heatmap(all_merged_data, f"Air_Pred_Diff_Deg{degree}", f"Air_Deviation_Deg{degree}")
    plot_mass_balance_violation(all_merged_data, min_order, max_order)

# Reorder dataframe and sort by the 'Mass_Flow' column
all_merged_data = all_merged_data[columns_order]
all_merged_data.sort_values(by='Mass_Flow', inplace=True)

# Export the total dataset
total_dataset_path = os.path.join(output_dir, f'Data_analysis_{current_date}.xlsx')
export_to_excel(total_dataset_path, {"Data_analysis": all_merged_data})

output_txt_path = os.path.join(output_dir, "Polynomial_Equations_Unconstrained.txt")

export_unconstrained_equations(
    models=models,
    polynomials=polynomials,
    metrics=metrics,
    feature_names=None,  # Optional, not used in current export
    output_path=output_txt_path
)

# STRESS TEST PART (DONE)
if run_stress_test:
    stress_test(models, polynomials, output_dir, min_order, max_order, current_date)
else:
    print("Skipping Stress Test!")
