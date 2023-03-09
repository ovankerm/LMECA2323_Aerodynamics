import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve

gamma = 1.4

def Prandtl(M):
    return np.sqrt((gamma + 1)/(gamma - 1)) * np.arctan(np.sqrt((gamma - 1)*(M*M - 1)/(gamma + 1))) - np.arctan(np.sqrt(M*M - 1))

def inverse_Prandtl(angle):
    func = lambda M : np.sqrt((gamma + 1)/(gamma - 1)) * np.arctan(np.sqrt((gamma - 1)*(M*M - 1)/(gamma + 1))) - np.arctan(np.sqrt(M*M - 1)) - angle
    return fsolve(func, 3)[0]

def get_beta(M1, theta):
    func = lambda beta : 1/(np.tan(beta) * ((gamma + 1)/2 * np.power(np.power(np.sin(beta), 2) - 1/(M1**2), -1) - 1)) - np.tan(theta)
    return fsolve(func, theta)[0]

def compute_values(h_c, angle, dx):
    N_panels = len(h_c) - 1

    angles = np.zeros(N_panels)
    Prandtls = np.zeros(N_panels)
    Ms = np.zeros(N_panels + 1)
    pressures = np.zeros(N_panels + 1)

    for i in range(N_panels):
        angles[i] = -np.arctan((h_c[i+1] - h_c[i])/dx)

    angles += np.radians(angle)

    M0 = 3.3
    theta_in = -angles[0]
    beta_in = get_beta(M0, theta_in)
    M1n = np.sqrt((2 + (gamma - 1) * np.power(M0 * np.sin(beta_in), 2))/(2 * gamma * np.power(M0 * np.sin(beta_in), 2) - (gamma - 1)))
    Ms[0] = M1n/np.sin(beta_in - theta_in)
    Prandtls[0] = Prandtl(Ms[0])
    pressures[0] = 1 + 2 * gamma * 1/(gamma + 1) * (np.power(M0 * np.sin(beta_in), 2) - 1) 

    for i in range(1, N_panels):
        Prandtls[i] = angles[i] - angles[0] + Prandtls[0]
        Ms[i] = inverse_Prandtl(Prandtls[i])
        pressures[i] = pressures[i-1] * np.power((1+(gamma-1)/2 * Ms[i-1] * Ms[i-1])/(1+(gamma-1)/2 * Ms[i] * Ms[i]), gamma/(gamma-1))

    theta_out = angles[-1]
    beta_out = get_beta(Ms[-2], theta_out)
    M_out_n = np.sqrt((2 + (gamma - 1) * np.power(Ms[-2] * np.sin(beta_out), 2))/(2 * gamma * np.power(Ms[-2] * np.sin(beta_out), 2) - (gamma - 1)))
    Ms[-1] = M_out_n/np.sin(beta_out - theta_out)
    pressures[-1] = pressures[-2] * (1 + 2 * gamma * 1/(gamma + 1) * (np.power(Ms[-2] * np.sin(beta_out), 2) - 1))

    return (pressures, Ms)


N_points = 65
x_c, dx = np.linspace(0, 1, endpoint=True, retstep=True, num=N_points)
h_c = 4 * 0.05 * x_c * (1 - x_c)

fig, ax = plt.subplots()
ax.plot(x_c, h_c, 'black')
ax.plot(x_c, -h_c, 'black')
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-1, 1)

pressures_0, Ms_0 = compute_values(h_c, 0, dx)
pressures_5, Ms_5 = compute_values(h_c, 5, dx)
pressures_10, Ms_10 = compute_values(h_c, 10, dx)

fig, ax = plt.subplots()
ax.plot(pressures_0, label='0')
ax.plot(pressures_5, label='5')
ax.plot(pressures_10, label='10')
ax.legend()
# ax.plot(pressures_bottom)
plt.show()