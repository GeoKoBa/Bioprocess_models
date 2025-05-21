import numpy as np
import matplotlib.pyplot as plt

# Constants (assumed values)
mu_max = 0.5      # Maximum specific growth rate for carbon source (1/h)
K_S = 1.0         # Half-saturation constant for carbon source (g/L)

Y_X_S = 0.4       # Yield coefficient of biomass on the carbon source (g biomass/g carbon source)
Y_P_X = 0.2       # Yield coefficient of product on biomass (g product/g biomass)

# Reactor parameters
total_volume = 100  # Total volume of the fermenter (L)
max_volume = 0.6 * total_volume  # Maximum operational volume (60% of total volume)
initial_volume = 0.4 * total_volume  # Initial volume (40% of total volume)

# Feed settings
feed_rate_L = 0.5   # Feed rate (L/h)
feed_conc_S = 10.0   # Concentration of carbon source in the feed (g/L)

# Initial conditions
C_S0 = 10.0        # Initial concentration of carbon source (g/L)
X0 = 0.1           # Initial biomass concentration (g/L)
P0 = 0.0           # Initial product concentration (g/L)
V0 = initial_volume  # Initial volume (L)

# Time settings
t_max = 50        # Maximum time (hours)
dt = 0.25         # Time step (hours)
time = np.arange(0, t_max + dt, dt)  # Time vector

# Arrays to store results
C_S_array = []
X_array = []
P_array = []
V_array = []

# Initial conditions
C_S = C_S0
X = X0
P = P0
V = V0

# Simulation loop
for t in time:
    # Store current values
    C_S_array.append(C_S)
    X_array.append(X)
    P_array.append(P)
    V_array.append(V)
    
    # Monod growth rate
    mu = (mu_max * C_S) / (K_S + C_S)  # Growth rate from carbon
    
    # Substrate consumption rate (Monod + dilution from feed)
    dS_dt = -(1 / Y_X_S) * mu * X + (feed_rate_L / V) * (feed_conc_S - C_S)
    
    # Biomass production rate
    dX_dt = mu * X
    
    # Product formation (growth-associated product formation)
    dP_dt = Y_P_X * mu * X
    
    # Volume increase due to feed rate
    dV_dt = feed_rate_L
    
    # Update variables (Euler integration)
    C_S += dS_dt * dt
    X += dX_dt * dt
    P += dP_dt * dt
    V += dV_dt * dt

# Plot results
plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(time, C_S_array, label='Carbon Source (C_S)')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration (g/L)')
plt.title('Carbon Source Concentration')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(time, X_array, label='Biomass (X)')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration (g/L)')
plt.title('Biomass Concentration')
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(time, P_array, label='Product (P)')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration (g/L)')
plt.title('Product Concentration')
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(time, V_array, label='Volume (V)')
plt.xlabel('Time (hours)')
plt.ylabel('Volume (L)')
plt.title('Reactor Volume')
plt.legend()

plt.tight_layout()
plt.show()
