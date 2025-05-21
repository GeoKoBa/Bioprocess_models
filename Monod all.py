import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Common parameters
S0 = 15.0  # Starting substrate concentration [g/L]
X0 = 0.2   # Inoculation biomass concentration [g/L]
Ks = 2.0   # Monod constant [g/L]
mumax = 0.3 # Maximum growth rate [1/h]
Yxs = 0.5  # Biomass/substrate yield [g/g]
qpmax = 0.4 # Max product rate [1/h]
Yps = 0.4  # Product/substrate yield [g/g]
Kp = 1.0   # Product constant [g/L]

# Additional parameters for Fed-Batch and Continuous
Sf = 10.0  # Substrate concentration in feed [g/L]
F = 0.05   # Feed rate [L/h]
V0 = 1.0   # Initial reactor volume [L]
D = 0.14   # Dilution rate [1/h]

# Kinetic Equations
def mu(S):
    return mumax * S / (Ks + S)

def qp(S):
    return qpmax * S / (Kp + S)

def Xn(X, S):
    return mu(S) * X

def Pn(X, S):
    return qp(S) * X

# Batch Model ODEs
def ODE_batch(t, x):
    X, P, S = x
    dX = Xn(X, S)
    dP = Pn(X, S)
    dS = (-1/Yxs) * Xn(X, S) + (-1/Yps) * Pn(X, S)
    return [dX, dP, dS]

# Fed-Batch Model ODEs
def ODE_fedbatch(t, x):
    X, P, S, V = x
    dX = Xn(X, S) - (F/V) * X
    dP = Pn(X, S) - (F/V) * P
    dS = (-1/Yxs) * Xn(X, S) + (F/V) * (Sf - S) + (-1/Yps) * Pn(X, S)
    dV = F
    return [dX, dP, dS, dV]

# Continuous (CSTR) Model ODEs
def ODE_continuous(t, x):
    X, S, P = x
    dX = Xn(X, S) - D * X
    dS = (-1/Yxs) * Xn(X, S) + D * (Sf - S) + (-1/Yps) * Pn(X, S)
    dP = Pn(X, S) - D * P
    return [dX, dS, dP]

# Function to run simulation
def run_simulation(operation_type):
    if operation_type == "Batch":
        y0 = [X0, 0, S0]  # Initial conditions for Batch
        t_span = (0, 50)
        sol = solve_ivp(ODE_batch, t_span, y0, t_eval=np.linspace(0, 50, 500))
        labels = ['Cell concentration', 'Product concentration', 'Substrate concentration']
    
    elif operation_type == "FedBatch":
        y0 = [X0, 0, S0, V0]  # Initial conditions for Fed-Batch
        t_span = (0, 50)
        sol = solve_ivp(ODE_fedbatch, t_span, y0, t_eval=np.linspace(0, 50, 500))
        labels = ['Cell concentration', 'Product concentration', 'Substrate concentration', 'Volume']
    
    elif operation_type == "CSTR":
        y0 = [X0, S0, 0]  # Initial conditions for Continuous
        t_span = (0, 50)
        sol = solve_ivp(ODE_continuous, t_span, y0, t_eval=np.linspace(0, 50, 500))
        labels = ['Cell concentration', 'Substrate concentration', 'Product concentration']
    
    else:
        print("Invalid operation type selected!")
        return
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    for i in range(len(sol.y)):
        plt.plot(sol.t, sol.y[i], label=labels[i])
    
    plt.xlabel('Duration [h]')
    plt.ylabel('Concentration [g/L]')
    plt.title(f'{operation_type} Fermentation Simulation')
    plt.legend()
    plt.show()

# Main Script
if __name__ == "__main__":
    print("Select the type of fermentation operation:")
    print("1. Batch")
    print("2. FedBatch")
    print("3. CSTR (Continuous)")
    
    choice = input("Enter the number corresponding to the operation type: ")
    operation_type = ""
    
    if choice == "1":
        operation_type = "Batch"
    elif choice == "2":
        operation_type = "FedBatch"
    elif choice == "3":
        operation_type = "CSTR"
    else:
        print("Invalid selection. Please choose 1, 2, or 3.")

    if operation_type:
        run_simulation(operation_type)
