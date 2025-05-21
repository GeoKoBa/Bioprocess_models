import numpy as np
import matplotlib.pyplot as plt

# Constants (assumed values)
mu_max = 0.5      # Maximum specific growth rate for carbon source (1/h)
K_S = 1.0         # Half-saturation constant for carbon source (g/L)
K_SI = 10.0       # Inhibition constant for carbon source (g/L)

v_N_max = 0.4     # Maximum specific consumption rate for nitrogen source (1/h)
K_N = 0.5         # Half-saturation constant for nitrogen source (g/L)
K_NI = 5.0        # Inhibition constant for nitrogen source (g/L)

Y_X_S = 0.4       # Yield coefficient of biomass on the carbon source (g biomass/g carbon source)

q_P_max = 0.3     # Maximum specific production rate of the product (1/h)
K_P = 1.0         # Half-saturation constant for product formation (g/L)
K_PI = 10.0       # Inhibition constant for product formation (g/L)

# Decision variables for inhibition (True or False)
inhibit_C = False  # Inhibit carbon source consumption
inhibit_N = True   # Inhibit nitrogen source consumption
inhibit_B = False  # Inhibit biomass production (usually not used directly)
inhibit_P = True   # Inhibit product production

# Reactor parameters
total_volume = 100  # Total volume of the fermenter (L)
max_volume = total_volume * 0.6  # Volume remains constant in batch mode

# Initial conditions
C_S = 10.0        # Initial concentration of carbon source (g/L)
C_N = 2.0         # Initial concentration of nitrogen source (g/L)
X = 0.0           # Initial biomass concentration (g/L)
P = 0.0           # Initial product concentration (g/L)
V = max_volume    # Initial volume (L)

# Time settings
t_max = 50        # Maximum time (hours)
dt = 0.25         # Time step (hours)
time = np.arange(0, t_max, dt)  # Time vector

# Arrays to store results
C_S_array = []
C_N_array = []
X_array = []
P_array = []

# Variables to track the total amount of carbon and nitrogen source consumed
total_consumed_C_S = 0.0
total_consumed_C_N = 0.0

# Simulation loop
for t in time:
    # Store current values
    C_S_array.append(C_S)
    C_N_array.append(C_N)
    X_array.append(X)
    P_array.append(P)
    
    # Rates of consumption and production with optional inhibition
    if inhibit_C:
        r_S = (mu_max * C_S) / (K_S + C_S + (C_S**2) / K_SI)
    else:
        r_S = (mu_max * C_S) / (K_S + C_S)
        
    if inhibit_N:
        r_N = (v_N_max * C_N) / (K_N + C_N + (C_N**2) / K_NI)
    else:
        r_N = (v_N_max * C_N) / (K_N + C_N)
        
    r_X = r_S * Y_X_S
    
    if inhibit_P:
        r_P = (q_P_max * C_S) / (K_P + C_S + (C_S**2) / K_PI)
    else:
        r_P = (q_P_max * C_S) / (K_P + C_S)
    
    # Update concentrations
    C_S -= r_S * dt
    C_N -= r_N * dt
    X += r_X * dt
    P += r_P * dt
    
    # Accumulate the total amount of carbon and nitrogen source consumed
    total_consumed_C_S += r_S * dt * max_volume
    total_consumed_C_N += r_N * dt * max_volume

# Convert final concentrations to grams
final_C_S = C_S * max_volume
final_C_N = C_N * max_volume
final_X = X * max_volume
final_P = P * max_volume

# Print final values
print(f"Final values for Batch Model:")
print(f"Total Carbon Source Consumed: {total_consumed_C_S:.2f} g")
print(f"Total Nitrogen Source Consumed: {total_consumed_C_N:.2f} g")
print(f"Total Carbon Source (C_S): {final_C_S:.2f} g")
print(f"Total Nitrogen Source (C_N): {final_C_N:.2f} g")
print(f"Total Biomass (X): {final_X:.2f} g")
print(f"Total Product (P): {final_P:.2f} g")


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

