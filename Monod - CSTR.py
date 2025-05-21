import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants and parameters
Q_IN = 5.0  # Inlet flow rate (L/h)
Q_OUT = 5.0  # Outlet flow rate (L/h)
V = 60.0  # Reactor volume (L)

mu_1 = 0.5      # Specific growth rate for biomass (1/h)
mu_2 = 0.2      # Specific rate for substrate (S) consumption (1/h)
mu_3 = 0.1      # Specific rate for product (P) production (1/h)
Y_S_X = 0.3     # Yield coefficient for substrate on biomass
Y_S_P = 0.1     # Yield coefficient for substrate on product
Y_N_X = 0.1     # Yield coefficient for nitrogen on biomass
K_P = 0.05      # Rate constant for product (P) production

# Feed concentrations (g/L)
X_F = 0.0       # Biomass 
S_F = 10.0      # Substrate
N_F = 2.0       # Nitrogen
P_F = 0.0       # Product

# Initial concentrations (g/L)
X0 = 0.1  # Initial biomass concentration
S0 = 10.0  # Initial substrate concentration
N0 = 2.0  # Initial nitrogen concentration
P0 = 0.1  # Initial product concentration

# Initial volume
V0 = V

# Time settings
t_max = 250  # Maximum time (hours)
dt = 0.1  # Time step (hours)
time = np.arange(0, t_max, dt)

# Differential equations
def cstr_odes(y, t):
    X, S, N, P, V = y
    
    dVdt = Q_IN - Q_OUT
    dXdt = mu_1 * X + (Q_IN * X_F - Q_OUT * X - X * dVdt) / V
    dSdt = (-Y_S_X * mu_2 * X - Y_S_P * mu_3 * X) + (Q_IN * S_F - Q_OUT * S - S * dVdt) / V
    dNdt = -Y_N_X * mu_1 * X + (Q_IN * N_F - Q_OUT * N - N * dVdt) / V
    dPdt = (mu_3 + K_P * S) * X + (Q_IN * P_F - Q_OUT * P - P * dVdt) / V
    
    return [dXdt, dSdt, dNdt, dPdt, dVdt]

# Initial conditions
initial_conditions = [X0, S0, N0, P0, V0]

# Integrate ODEs
results = odeint(cstr_odes, initial_conditions, time)

# Extract results
X = results[:, 0]
S = results[:, 1]
N = results[:, 2]
P = results[:, 3]
V = results[:, 4]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(time, X, label='Biomass (X)')
plt.plot(time, S, label='Substrate (S)')
plt.plot(time, N, label='Nitrogen (N)')
plt.plot(time, P, label='Product (P)')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration (g/L)')
plt.title('Concentration Profiles in CSTR')
plt.legend()
plt.grid(True)
plt.show()
