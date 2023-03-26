import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve

gamma = 1.4

def Prandtl(M):
    return np.sqrt((gamma + 1)/(gamma - 1)) * np.arctan(np.sqrt((gamma - 1)*(M*M - 1)/(gamma + 1))) - np.arctan(np.sqrt(M*M - 1))

def inverse_Prandtl(angle, init):
    func = lambda M : np.sqrt((gamma + 1)/(gamma - 1)) * np.arctan(np.sqrt((gamma - 1)*(M*M - 1)/(gamma + 1))) - np.arctan(np.sqrt(M*M - 1)) - angle
    return fsolve(func, init)[0]

def get_beta(M1, theta):
    func = lambda beta : 1/(np.tan(beta) * ((gamma + 1)/2 * np.power(np.power(np.sin(beta), 2) - 1/(M1**2), -1) - 1)) - np.tan(theta)
    return fsolve(func, theta)[0]

def expansion_fan(M1, turn):
    M2 = inverse_Prandtl(turn + Prandtl(M1), M1)
    return (M2, np.power((1+(gamma-1)/2 * M1 * M1)/(1+(gamma-1)/2 * M2 * M2), gamma/(gamma-1)))

def oblique_shock(M1, turn):
    beta_in = get_beta(M1, turn)
    M1n = np.sqrt((2 + (gamma - 1) * np.power(M1 * np.sin(beta_in), 2))/(2 * gamma * np.power(M1 * np.sin(beta_in), 2) - (gamma - 1)))
    return  (M1n/np.sin(beta_in - turn), 1 + 2 * gamma * 1/(gamma + 1) * (np.power(M1 * np.sin(beta_in), 2) - 1))

def compute_values(h_c, angle, dx, M0, side):
    N = len(h_c)

    angles = np.zeros(N + 1)
    for i in range(1, N):
        angles[i] = -np.arctan((h_c[i] - h_c[i-1])/dx) + np.radians(angle)

    if side == 'bottom': angles *= -1

    Ms = np.zeros(N + 1)
    pressures = np.zeros(N + 1)

    Ms[0] = M0
    pressures[0] = 1

    CD = 0
    CL = 0

    for i in range(1, len(Ms)):
        turn = angles[i] - angles[i-1]
        if(turn > 0):
            Ms[i], p = expansion_fan(Ms[i-1], turn)
            pressures[i] = pressures[i-1] * p
        else:
            Ms[i], p = oblique_shock(Ms[i-1], -turn)
            pressures[i] = pressures[i-1] * p
        if(i < len(Ms) - 1):
            CD += 2/(gamma * M0 * M0) * dx * pressures[i] * np.tan(-angles[i])
            CL -= 2/(gamma * M0 * M0) * dx * pressures[i]

    if side == 'bottom': CL *= -1

    return (pressures, angles, CD, CL)

#---------- Question 2 ----------#
N_points = 65
x_c, dx = np.linspace(0, 1, endpoint=True, retstep=True, num=N_points)
mid_x = np.zeros(N_points - 1)
for i in range(N_points - 1):
    mid_x[i] = 0.5 * (x_c[i+1] + x_c[i])
h_c = 4 * 0.05 * x_c * (1 - x_c)
M1 = 3.3
pressure_top, angles_top, _, _ = compute_values(h_c, 0, dx, M1, 'top')
pressure_bottom, angles_bottom, _, _ = compute_values(-h_c, 0, dx, M1, 'bottom')
CP_top = 2/(gamma * M1 * M1) * (pressure_top - 1)
CP_bottom = 2/(gamma * M1 * M1) * (pressure_bottom - 1)
CP_thin_top = -2/np.sqrt(M1 * M1 - 1) * angles_top
CP_thin_bottom = -2/np.sqrt(M1 * M1 - 1) * angles_bottom


fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(mid_x, CP_bottom[1:-1], "k-", label=r"Calculated $C_P$")
ax.plot(mid_x, CP_thin_bottom[1:-1], "k--", label=r"$C_P$ obtained with the thin airfoil theory")
ax.legend()
ax.grid()
ax.set_xlabel('x/c')
ax.set_ylabel(r'$C_p$')
ax.set_title('Pressure coefficient')
ax.set_xticks([0.1 * i for i in range(11)])

plt.savefig("HW1_images/CP.eps", format='eps')

#---------- Question 3 ----------#
N_points = 65
x_c, dx = np.linspace(0, 1, endpoint=True, retstep=True, num=N_points)
h_c = 4 * 0.05 * x_c * (1 - x_c)

M1 = 3.3
angles = np.linspace(0, 15, endpoint=True, num=50)
CD_airfoil = np.zeros_like(angles)
CL_airfoil = np.zeros_like(angles)
for i, a in enumerate(angles):
    _, _, CD_top, CL_top = compute_values(h_c, a, dx, M1, 'top')
    _, _, CD_bottom, CL_bottom = compute_values(-h_c, a, dx, M1, 'bottom')
    CD_airfoil[i] = CD_top + CD_bottom
    CL_airfoil[i] = CL_top + CL_bottom

CL_thin_airfoil = 4 * np.radians(angles) * np.power(M1 * M1 - 1, -1/2)
CD_thin_airfoil = 4 * np.power(M1 * M1 - 1, -1/2) * (np.power(np.radians(angles), 2) + 16/3 * 0.05 * 0.05)

#---------- Question 4 ----------#
N_points = 3
x_c, dx = np.linspace(0, 1, endpoint=True, retstep=True, num=N_points)
h_c = 4 * 0.05 * x_c * (1 - x_c)

M1 = 3.3
angles = np.linspace(0, 15, endpoint=True, num=50)
CD_diamond = np.zeros_like(angles)
CL_diamond = np.zeros_like(angles)
for i, a in enumerate(angles):
    _, _, CD_top, CL_top = compute_values(h_c, a, dx, M1, 'top')
    _, _, CD_bottom, CL_bottom = compute_values(-h_c, a, dx, M1, 'bottom')
    CD_diamond[i] = CD_top + CD_bottom
    CL_diamond[i] = CL_top + CL_bottom

CL_thin_diamond = 4 * np.radians(angles) * np.power(M1 * M1 - 1, -1/2)
CD_thin_diamond = 4 * np.power(M1 * M1 - 1, -1/2) * (np.power(np.radians(angles), 2) + 4 * 0.05 * 0.05)


#---------- Plots ----------#
fig, ax = plt.subplots(figsize=(12, 7))
# ax.plot(angles, CD_thin_airfoil, 'k--', label=r"$C_D$ using thin airfoil theory")
# ax.plot(angles, CL_thin_airfoil, 'b--', label=r"$C_L$ using thin airfoil theory")

ax.plot(angles, CD_airfoil, 'k:', label=r"$C_D$ of the airfoil")
ax.plot(angles, CL_airfoil, 'b:', label=r"$C_L$ of the airfoil")
ax.plot(angles, CD_diamond, 'k-', label=r"$C_D$ of the diamond")
ax.plot(angles, CL_diamond, 'b-', label=r"$C_L$ of the diamond")
ax.plot(angles, CD_thin_diamond, 'k--', label=r"$C_D$ of the diamond using thin airfoil theory")
ax.plot(angles, CL_thin_diamond, 'b--', label=r"$C_L$ of the diamond using thin airfoil theory")

# ax.plot(angles, CD_diamond, label="CD diamond")
# ax.plot(angles, CL_diamond, label="CL diamond")
# ax.plot(angles, CD_thin_diamond, label="CD thin diamond")
# ax.plot(angles, CL_thin_diamond, label="CL thin diamond")
# ax.plot(angles, CL_airfoil/CD_airfoil, 'k-', label=r"$C_L/C_D$ for the airfoil")
# ax.plot(angles, CL_thin_airfoil/CD_thin_airfoil, 'k--', label=r"$C_L/C_D$ for the thin airfoil")
# ax.plot(angles, CL_diamond/CD_diamond, label="CL/CD diamond")
# ax.plot(angles, CL_thin_diamond/CD_thin_diamond, label="CL/CD thin diamond")
# ax.plot(CD_airfoil, CL_airfoil, label="CL/CD airfoil")
# ax.plot(CD_diamond, CL_diamond, label="CL/CD diamond")

# i_max = np.argmax(CL_airfoil/CD_airfoil)
# z = np.polyfit(angles[i_max-2: i_max+3], (CL_airfoil/CD_airfoil)[i_max-2: i_max+3], 2)
# a_max = -z[1]/(2 * z[0])
# print("max_angle :", a_max, "max_val :", z[0] * a_max**2 + z[1] * a_max + z[2])
# ax.scatter(a_max, z[0] * a_max**2 + z[1] * a_max + z[2], c='k', label='Max values')

# i_max = np.argmax(CL_thin_airfoil/CD_thin_airfoil)
# z = np.polyfit(angles[i_max-2: i_max+3], (CL_thin_airfoil/CD_thin_airfoil)[i_max-2: i_max+3], 2)
# a_max = -z[1]/(2 * z[0])
# print("max_angle :", a_max, "max_val :", z[0] * a_max**2 + z[1] * a_max + z[2])
# ax.scatter(a_max, z[0] * a_max**2 + z[1] * a_max + z[2], c='k')



ax.grid()
ax.legend()
ax.set_xlabel('Angle of attack [°]')
ax.set_ylabel(r'$C_{L, D}$')
ax.set_title('Lift to drag ratio depending of the angle of attack')
ax.set_xticks([i for i in range(16)])

fig.savefig("HW1_images/CL_CD_diamond.eps", format='eps')


# fig, ax = plt.subplots()
# ax.plot(x_c, h_c, 'black')
# ax.plot(x_c, -h_c, 'black')
# ax.set_xlim(-0.1, 1.1)
# ax.set_ylim(-1, 1)
plt.show()