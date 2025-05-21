import re

# Constants
# Yield constants
Yxs = 0.483        # Biomass yield from oxidative carbon source consumption [g_biom/g_subs]
Yps = 7.06e-2      # Product yield from substrate [g_prod/g_subs]

# Fixed reactor and operational parameters
V_total = 10                                        # Fixed total reactor volume [L]
percent_operational = 0.60                          # Fixed operational percentage (e.g., 60%)
V_effective = V_total * percent_operational         # Effective volume available for operation [L]
operation_days = 7                                  # Fixed number of operation days
operation_hours = operation_days * 24               # Operation hours based on fixed days
operation_minutes = operation_hours * 60            # Operation hours based on fixed days

# Biomass composition - C_1 H_1.81 O_0.52 N_0.21 - Another source was C_1 H_1.73 O_0.47 N_0.2
HX = 1.81                                           # Hydrogen content of biomass [mol/C-mol]
OX = 0.52                                           # Oxygen content of biomass [mol/C-mol]
NX = 0.21                                           # Nitrogen content of biomass [mol/C-mol]

# Molecular & Atomic weights of elements
MW_O2 = 32                                          # Oxygen [g/mol]
MW_CO2 = 44                                         # Carbon Dioxide [g/mol]
MW_C = 12.01                                        # Carbon [g/mol]
MW_H = 1.008                                        # Hydrogen [g/mol]
MW_O = 16.00                                        # Oxygen [g/mol]
MW_N = 14.01                                        # Nitrogen [g/mol]
MW_X = 12 + HX + OX * 16 + NX * 14                  # Biomass molecular weight [g/C-mol]

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
c = Yxs * MW_S / MW_X   # Biomass coefficient
d = w - c - f * j
b = (c * NX + f * m - z) / i
e = (x + b * g - f * k - c * HX) / 2
a = (c * OX + 2 * d + e + f * l - y - b * h) / 2

print(f"a = {a:.4f}")
print(f"b = {b:.4f}")
print(f"c = {c:.4f}")
print(f"d = {d:.4f}")
print(f"e = {e:.4f}")
print(f"f = {f:.4f}")

# Print final chemical reaction
print("\nFinal chemical reaction achieved:")
print(f"C{w}H{x}O{y}N{z} + {a:.4f} O2 + {b:.4f} {nitrogen_source_formula} -> {c:.4f} CH{HX}O{OX}N{NX} + {d:.4f} CO2 + {e:.4f} H2O + {f:.4f} C{j}H{k}O{l}N{m}")

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

    # Print Mass Balance
    print("\nMass Balance Check:")
    print(f"Carbon balance: Reactants = {C_reactants:.4f}, Products = {C_products:.4f}, Difference = {C_diff:.4f}")
    print(f"Hydrogen balance: Reactants = {H_reactants:.4f}, Products = {H_products:.4f}, Difference = {H_diff:.4f}")
    print(f"Oxygen balance: Reactants = {O_reactants:.4f}, Products = {O_products:.4f}, Difference = {O_diff:.4f}")
    print(f"Nitrogen balance: Reactants = {N_reactants:.4f}, Products = {N_products:.4f}, Difference = {N_diff:.4f}")

mass_balance()

# Production rates function
def calculate_production_parameters(initial_biomass_concentration, final_biomass_concentration):
    net_biomass_concentration = final_biomass_concentration - initial_biomass_concentration

    # Oxygen Uptake Rate (OUR) in mmol/h based on net biomass production
    OUR = a * MW_O2 * net_biomass_concentration / MW_S
    print(f"Oxygen Uptake Rate (OUR): {OUR:.4f} mmol/h")

    # Product - production rate in g/h based on net biomass concentration
    product_production_rate = f * net_biomass_concentration  # Product production rate in g/h
    print(f"Product production rate: {product_production_rate:.4f} g/h")

    # Product - total production over the operation time
    total_product_produced = product_production_rate * operation_hours  # Total product produced in g
    print(f"Total product produced: {total_product_produced:.4f} g")

    # Product - production rate per total reactor volume (g/L)
    product_production_rate_per_volume = total_product_produced / V_total  # Product in g/L
    print(f"Product production rate per total reactor volume: {product_production_rate_per_volume:.4f} g/L")

    # Biomass Production Rate in g/h
    biomass_production_rate = c * MW_X * net_biomass_concentration / MW_S
    print(f"Biomass Production Rate: {biomass_production_rate:.4f} g/h")

    # Total Carbon Source Consumption in g
    total_Csource_consumption = q_Csource * V_effective * operation_hours
    print(f"Total C-Source Consumption: {total_Csource_consumption:.4f} g")

    # Calculate moles of CO2 produced
    moles_CO2 = d * net_biomass_concentration / MW_S

    # Constants for ideal gas law
    R = 0.0821  # Ideal gas constant in L·atm/(mol·K)
    T = 313.15  # Temperature in Kelvin (40°C)
    P = 1  # Pressure in atmospheres

    # Calculate volume of CO2 produced
    volume_CO2 = (moles_CO2 * R * T) / P
    print(f"Volume of CO2 produced: {volume_CO2:.4f} liters")

    # Calculate flow rate of CO2
    flow_rate_CO2 = volume_CO2 / operation_minutes * 2 #To accound for sampling 120 wiffs every other hour
    print(f"Flow rate of CO2: {flow_rate_CO2:.4f} liters/min")

initial_biomass_concentration = float(input("Enter initial biomass concentration (g/L): "))
final_biomass_concentration = float(input("Enter final biomass concentration (g/L): "))
calculate_production_parameters(initial_biomass_concentration, final_biomass_concentration)

