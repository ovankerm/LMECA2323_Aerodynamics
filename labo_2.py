import numpy as np
from matplotlib import pyplot as plt

g = 9.81
c_bar = 0.05
b = 0.3
S = b * c_bar
rho = 1.225
mu = 1.81e-5
nu = mu/rho
C_D_arms = 0.06433
A_r = b/c_bar
print("Aspect ratio Ar =", A_r)

p = 313 # [Pa]
U_inf = np.sqrt(2 * p/rho)
Re = U_inf * c_bar/nu
print("U_inf =", U_inf, "m/s")
print("Re = %e"%Re)

D_offset = 0.461
M_D_calib=np.array([0,0.06816,0.181558,0.294956,0.408354,0.521752,0.748548,0.975344])
V_D_calib=np.array([0.71,1.02,1.54,2.07,2.59,3.14,4.2,5.24])
F_D_calib = M_D_calib * g
fit_D = np.polyfit(V_D_calib, F_D_calib, 1)
tension_to_drag = lambda V : fit_D[0] * (V - D_offset)

print(f"D = {fit_D[0]} (V - {D_offset})")

L_offset = -0.192
M_L_calib=np.array([0,0.06816,0.181558,0.294956,0.408354,0.521752])
V_L_calib=np.array([-0.2,-0.16,-0.1,-0.04,0.02,0.14])
F_L_calib = M_L_calib * g
fit_L = np.polyfit(V_L_calib, F_L_calib, 1)
tension_to_lift = lambda V : fit_L[0] * (V - L_offset)

print(f"L = {fit_L[0]} (V - {L_offset})")

angles = np.array([i for i in range(-8, 21, 2)])

L_tension = np.array([-0.34, -0.291, -0.239, -0.222, -0.191, -0.165, -0.124, -0.061, -0.027, -0.030, -0.037, -0.053, -0.065, -0.077, -0.077])
D_tension = np.array([0.748, 0.700, 0.682, 0.685, 0.684, 0.686, 0.696, 0.747, 0.860, 0.920, 0.955, 1.026, 1.085, 1.138, 1.194])

C_L = tension_to_lift(L_tension)/(p * S)
C_D = tension_to_drag(D_tension)/(p * S) - C_D_arms

fit_polar = np.poly1d(np.polyfit(C_L[2:8], C_D[2:8], 2))
a = np.linspace(-0.45, 0.45)

max = np.max(C_L/C_D)
print("CL/CD max = ", max)

e = 1/(np.pi * A_r * fit_polar.c[0])
print("Oswald efficiency e =", e)

fig, ax = plt.subplots()
ax.grid()
ax.plot(angles, C_L, 'k')
ax.set_xlabel("Angle of attack [°]")
ax.set_ylabel(r"$C_L$ [-]")
ax.set_title(r"Lift coefficient : $C_L$")
fig.savefig("lab_2_images/CL.eps", format = 'eps')

fig, ax = plt.subplots()
ax.grid()
ax.plot(angles, C_D, 'k')
ax.set_xlabel("Angle of attack [°]")
ax.set_ylabel(r"$C_D$ [-]")
ax.set_title(r"Drag coefficient : $C_D$")
fig.savefig("lab_2_images/CD.eps", format = 'eps')


x = np.linspace(0, 0.1)
i = np.argmax(C_L/C_D)
fig, ax = plt.subplots()
ax.grid()
ax.plot(C_D, C_L, 'k', label='Experimental values')
ax.plot(fit_polar(a), a, 'b', label='Second order fit')
ax.plot(x, max * x, 'k-.')
ax.scatter(C_D[i], C_L[i], c='r', label = r"max$(\frac{C_L}{C_D})$")
ax.set_xlabel(r"$C_D$ [-]")
ax.set_ylabel(r"$C_L$ [-]")
ax.set_title(r"Wing polar")
ax.legend()
fig.savefig("lab_2_images/polar.eps", format = 'eps')


plt.show()

