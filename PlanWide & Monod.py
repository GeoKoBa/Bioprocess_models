import re
import numpy as np
import matplotlib.pyplot as plt
import math

# FUNCTIONS
# Mass Balance Check Function
def mass_balance():
    # Reactants
    C_reactants = w                             # Carbon from the carbon source and 0 from other reactants
    H_reactants = x + b * g                     # Hydrogen from carbon source and nitrogen source
    O_reactants = y + a * 2 + h                 # Oxygen from carbon source, O2, and nitrogen source
    N_reactants = z + i * b                     # Nitrogen from nitrogen source

    # Products
    C_products = c + d * 1 + f * j              # Carbon from biomass, CO2, and the product
    H_products = c * HX + e * 2 + f * k         # Hydrogen from biomass, H2O, and the product
    O_products = c * OX + d * 2 + e + f * l     # Oxygen from biomass, CO2, H2O, and the product
    N_products = c * NX + f * m                 # Nitrogen from biomass and the product

    # Calculate differences
    C_diff = C_reactants - C_products
    H_diff = H_reactants - H_products
    O_diff = O_reactants - O_products
    N_diff = N_reactants - N_products
    MB_check = abs(C_diff) + abs(H_diff) + abs(O_diff) + abs(N_diff)

    # Print Mass Balance
    if MB_check > 0.0000:
        print("\nMass Balance Error:")
        print(f"Carbon balance: Reactants = {C_reactants:.4f}, Products = {C_products:.4f}, Difference = {C_diff:.4f}")
        print(f"Hydrogen balance: Reactants = {H_reactants:.4f}, Products = {H_products:.4f}, Difference = {H_diff:.4f}")
        print(f"Oxygen balance: Reactants = {O_reactants:.4f}, Products = {O_products:.4f}, Difference = {O_diff:.4f}")
        print(f"Nitrogen balance: Reactants = {N_reactants:.4f}, Products = {N_products:.4f}, Difference = {N_diff:.4f}")
    else:
        print("\nMass Balance within tolerance.")

# VARIABLES
# Yield
Yxs = 0.483                                 # Biomass yield from oxidative carbon source consumption [g_biom/g_subs]
Yps = 7.06e-2                               # Product yield from substrate [g_prod/g_subs]
mu_max = 0.5                                # Maximum specific growth rate for carbon source (1/h)
# Monod
K_S = 1.0                                   # Half-saturation constant for carbon source (g/L)
K_SI = 10.0                                 # Inhibition constant for carbon source (g/L)
c_N_max = 0.4                               # Maximum specific consumption rate for nitrogen source (1/h)
K_N = 0.5                                   # Half-saturation constant for nitrogen source (g/L)
K_NI = 5.0                                  # Inhibition constant for nitrogen source (g/L)
p_P_max = 0.3                               # Maximum specific production rate of the product (1/h)
K_P = 1.0                                   # Half-saturation constant for product formation (g/L)
K_PI = 10.0                                 # Inhibition constant for product formation (g/L)
# Decision variables for inhibition (True or False)
inhibit_C = False                           # Inhibit carbon source consumption
inhibit_N = True                            # Inhibit nitrogen source consumption
inhibit_B = False                           # Inhibit biomass production (usually not used directly)
inhibit_P = True                            # Inhibit product production
# Molecular & Atomic weights of elements
MW_O2 = 32                                  # Oxygen [g/mol]
MW_CO2 = 44                                 # Carbon Dioxide [g/mol]
MW_C = 12.01                                # Carbon [g/mol]
MW_H = 1.008                                # Hydrogen [g/mol]
MW_O = 16.00                                # Oxygen [g/mol]
MW_N = 14.01                                # Nitrogen [g/mol]
# Reactor and operational parameters
V_total = 100                               # Fixed total reactor volume [L]
percent_operational = 0.75                  # Fixed operational percentage (e.g., 60%)
V_eff = V_total * percent_operational       # Effective volume available for operation [L]
d = 1                                       # Reactor diameter [m]
A = np.pi/4*d**2;                           # height [m]
h = (V_eff/1000)/A;
# Operation parameters
operation_days = 7                          # Fixed number of operation days
operation_hours = operation_days * 24       # Operation hours based on fixed days
operation_minutes = operation_hours * 60    # Operation hours based on fixed days
dt = 0.25                                   # Time step (hours)
time = np.arange(0, operation_hours, dt)    # Time vector
#Concentration
C_S = 50.0                                  # Initial concentration of carbon source (g/L)
C_N = 15.0                                  # Initial concentration of nitrogen source (g/L)
C_X = 0.0                                   # Initial biomass concentration (g/L)
C_P = 0.0                                   # Initial product concentration (g/L)
# Biomass composition - C_1 H_1.81 O_0.52 N_0.21 - Another source was C_1 H_1.73 O_0.47 N_0.2
HX = 1.81                                   # Hydrogen content of biomass [mol/C-mol]
OX = 0.52                                   # Oxygen content of biomass [mol/C-mol]
NX = 0.21                                   # Nitrogen content of biomass [mol/C-mol]
MW_X = 12 + HX + OX * 16 + NX * 14          # Biomass molecular weight [g/C-mol]

# List of carbon sources and their properties
carbon_sources = {
    "1": {"name": "Glucose", "MW": 180.16, "uptake_rate": 0.1, "formula": "C6H12O6", "w": 6, "x": 12, "y": 6, "z": 0},
    "2": {"name": "Sucrose", "MW": 342.30, "uptake_rate": 0.1, "formula": "C12H22O11", "w": 12, "x": 22, "y": 11, "z": 0},
    "3": {"name": "Glycerol", "MW": 92.09, "uptake_rate": 0.1, "formula": "C3H8O3", "w": 3, "x": 8, "y": 3, "z": 0},
    "4": {"name": "Lactose", "MW": 342.297, "uptake_rate": 0.1, "formula": "C12H22O11", "w": 12, "x": 22, "y": 11, "z": 0},
    "5": {"name": "Mannitol", "MW": 182.17, "uptake_rate": 0.1, "formula": "C6H14O6", "w": 6, "x": 14, "y": 6, "z": 0}
}

# Display carbon source options
print("Select a carbon source:")
for key, source in carbon_sources.items():
    print(f"{key}: {source['name']}")

# Select carbon source
selection = input("Select C-source: ")
if selection not in carbon_sources:
    print("Invalid selection. Using default (Glucose).")
    selection = "1"

# Get selected carbon source properties
selected_source = carbon_sources[selection]
MW_S = selected_source["MW"]
q_Csource = selected_source["uptake_rate"]
Csource_formula = selected_source["formula"]

# Define w, x, y, z based on the selected carbon source
w = selected_source["w"]
x = selected_source["x"]
y = selected_source["y"]
z = selected_source["z"]

# Select Nitrogen Source 
nitrogen_source_formula = input("Select N-source (formula H_gO_hN_i): ")

# Parse the chemical composition of the nitrogen source
matches = re.findall(r'([A-Z])(\d*)', nitrogen_source_formula)
nitrogen_composition = {elem: int(num) if num else 1 for elem, num in matches}

# Extract the number of atoms for H, O, N
g = nitrogen_composition.get('H', 0)  # Hydrogen coefficient
h = nitrogen_composition.get('O', 0)  # Oxygen coefficient
i = nitrogen_composition.get('N', 0)  # Nitrogen coefficient

# Calculate the molecular weight of the nitrogen source
MW_Ns = (
    g * MW_H +
    h * MW_O +
    i * MW_N
)

# Get desired product information from user
product_formula = input(f"Enter the chemical formula of desired product (e.g., C_jH_kO_lN_m): ")

# Parse the chemical composition of the product and calculate MW
matches = re.findall(r'([A-Z])(\d*)', product_formula)
product_composition = {elem: int(num) if num else 1 for elem, num in matches}

# Calculate MW of the product
MW_P = (
    product_composition.get('C', 0) * MW_C +
    product_composition.get('H', 0) * MW_H +
    product_composition.get('O', 0) * MW_O +
    product_composition.get('N', 0) * MW_N
)

# Get the number of atoms for C, H, O, N
j = product_composition.get('C', 0)
k = product_composition.get('H', 0)
l = product_composition.get('O', 0)
m = product_composition.get('N', 0)

# Calculate stoichiometric coefficients
f = Yps * MW_S / MW_P   # Product coefficient
# print(f"f = {f:.4f}")
c = Yxs * MW_S / MW_X   # Biomass coefficient
# print(f"c = {c:.4f}")
d = w - c - f * j
# print(f"d = {d:.4f}")
b = (c * NX + f * m - z) / i
# print(f"b = {b:.4f}")
e = (x + b * g - f * k - c * HX) / 2
# print(f"e = {e:.4f}")
a = (c * OX + 2 * d + e + f * l - y - b * h) / 2 
# print(f"a = {a:.4f}")

# Print final chemical reaction 
print("\nFinal chemical reaction achieved:")
print(f"C{w}H{x}O{y}N{z} + {a:.4f} O2 + {b:.4f} {nitrogen_source_formula} -> {c:.4f} CH{HX}O{OX}N{NX} + {d:.4f} CO2 + {e:.4f} H2O + {f:.4f} C{j}H{k}O{l}N{m}")

mass_balance()

# Arrays to store results
C_S_array = []
C_N_array = []
X_array = []
P_array = []

# Variables to track the total amount of carbon and nitrogen source consumed
total_consumed_C_S = 0.0
total_consumed_C_N = 0.0

n_CO2 = (V_eff * C_S)/MW_S * d

# Simulation loop
for t in time:
    # Store current values
    C_S_array.append(C_S)
    C_N_array.append(C_N)
    X_array.append(C_X) 
    P_array.append(C_P)
    
    # Rates of consumption and production with optional inhibition
    if inhibit_C:
        r_S = (mu_max * C_S) / (K_S + C_S + (C_S**2) / K_SI)
    else:
        r_S = (mu_max * C_S) / (K_S + C_S)
        
    if inhibit_N:
        r_N = (c_N_max * C_N) / (K_N + C_N + (C_N**2) / K_NI)
    else:
        r_N = (c_N_max * C_N) / (K_N + C_N)
        
    r_X = r_S * Yxs
    
    if inhibit_P:
        r_P = (p_P_max * C_S) / (K_P + C_S + (C_S**2) / K_PI)
    else:
        r_P = (p_P_max * C_S) / (K_P + C_S)
    
    # Update concentrations
    C_S -= r_S * dt
    C_N -= r_N * dt
    C_X += r_X * dt
    C_P += r_P * dt
    
    # Accumulate the total amount of carbon and nitrogen source consumed
    total_consumed_C_S += r_S * dt * V_eff
    total_consumed_C_N += r_N * dt * V_eff

# Convert final concentrations to grams
final_C_S = C_S * V_eff
final_C_N = C_N * V_eff
final_C_X = C_X * V_eff
final_C_P = C_P * V_eff

# Print final values
print(f"Final values for Batch Model:")
print(f"Total Carbon Source Consumed: {total_consumed_C_S:.2f} g")
print(f"Total Nitrogen Source Consumed: {total_consumed_C_N:.2f} g")
print(f"Final Carbon Source (C_S): {final_C_S:.2f} g")
print(f"Final Nitrogen Source (C_N): {final_C_N:.2f} g")
print(f"Final Biomass (X): {final_C_X:.2f} g")
print(f"Final Product (P): {final_C_P:.2f} g")


# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(time, C_S_array, label='Carbon Source (C_S)')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration (g/L)')
plt.title('Carbon Source Concentration')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(time, C_N_array, label='Nitrogen Source (C_N)')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration (g/L)')
plt.title('Nitrogen Source Concentration')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(time, X_array, label='Biomass (X)')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration (g/L)')
plt.title('Biomass Concentration')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(time, P_array, label='Product (P)')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration (g/L)')
plt.title('Product Concentration')
plt.legend()

plt.tight_layout()
plt.show()

# Production rates function

net_C_S = final_C_X - C_X
# Oxygen Uptake Rate (OUR) in mmol/h based on net biomass production
#OUR = ...
#print(f"Oxygen Uptake Rate (OUR): {OUR:.4f} mmol/h")
        
# Production rate based on net biomass concentration g/h                           
# P_rate = ...   
#print(f"Product production rate: {P_rate:.4f} g/h") 

# Total production over the operation time g                                               
#total_P = ...
#print(f"Total product produced: {total_P:.4f} g")                                     

# Production rate per total reactor volume g/L
#P_rate_V = ...
#print(f"Product production rate per total reactor volume: {P_rate_V:.4f} g/L")                                           

# Biomass Production Rate in g/h
#X_rate = ...                                    
#print(f"Biomass Production Rate: {X_rate:.4f} g/h")

# Total Carbon Source Consumption in g
#total_C_consumption = ...        
#print(f"Total C-Source Consumption: {total_C_consumption:.4f} g")

# Moles of CO2 produced
#moles_CO2 = d * net_C_X / MW_S                                         

# Constants for ideal gas law
R = 0.0821  # Ideal gas constant in L·atm/(mol·K)
T = 303.15  # Temperature in Kelvin (30°C)
P = 1  # Pressure in atmospheres
# Calculate volume of CO2 produced

volume_CO2 = (n_CO2 * R * T) / P
print(f"Volume of CO2 produced: {volume_CO2:.4f} liters")
# Calculate flow rate of CO2
flow_rate_CO2 = volume_CO2 / operation_minutes * 2 #To accound for sampling 120 wiffs every other hour
print(f"Flow rate of CO2: {flow_rate_CO2:.4f} liters/min")
