import numpy as np
from numpy.linalg import solve
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from regularized_step_function_students import *

def sin_sin(n, theta):
    return np.where(np.in1d(theta, [0, np.pi]), n, np.sin(n * theta)/np.sin(theta))

# CONSTANTS A380
a_0 = 2 * np.pi  # [-]
Lam = 0.3  # [-]
lam = (1-Lam)/(1+Lam)
A_r = 10  # [-]
b = 80  # [m]
U_inf = 900/3.6  # [m/s]
rho = 0.4135  # [km/m^3]
g = 9.81  # [m/s^2]
M = 575e3  # [kg]
tau = 0.05

# FUNCTIONS
c_norm = lambda xsi : (1 + lam) - 2 * lam * np.abs(xsi)
q = lambda theta, n : np.sin(n * theta) + 1/4 * a_0 * c_norm(np.cos(theta)) * n/A_r * sin_sin(n, theta)
r = lambda theta : a_0/2 * c_norm(np.cos(theta))/A_r

#------1------#
# Gamma(theta)
num = 30
A = np.zeros((num, num))
c = np.zeros(num)

theta_int = np.linspace(0, np.pi/2, num=50, endpoint=False)

for i in range(1, 2 * num + 1, 2):
    c[(i-1)//2] = np.trapz(r(theta_int) * q(theta_int, i), theta_int)
    for j in range(1, 2 * num + 1, 2):
        A[(i-1)//2, (j-1)//2] = np.trapz(q(theta_int, j) * q(theta_int, i), theta_int)

b_n = solve(A, c)

theta = np.linspace(0, np.pi, num = 101)
Gamma_norm = np.zeros_like(theta)
for i in range(num):
    Gamma_norm += b_n[i] * np.sin((2 * i + 1) * theta)


print("------1------")
# Lift
A_0 = np.pi/2 * A_r * b_n[0]
print("dCL/d(alpha) =", A_0)

f = lambda alpha : np.pi/4 * rho * U_inf*U_inf * b*b * b_n[0] * alpha * np.cos(alpha) - M*g

alpha_w = fsolve(f, np.radians(5))[0]

C_L = np.round(A_0 * alpha_w, 3)
alpha_w = C_L/A_0

print("C_L =", C_L)
print("alpha =", np.degrees(alpha_w), "°")

# Span loading
K = 2 * C_L * A_r/A_0 * Gamma_norm
fig, ax = plt.subplots()
ax.grid()
ax.plot(np.cos(theta), K, "k")
ax.set_xlabel(r"$\xi$")
ax.set_ylabel("K")
ax.set_title("Span loading")
fig.savefig("HW2_images/Span_loading_1.eps", format='eps')

# AoA
epsilon = np.zeros_like(theta)
for i in range(num):
    epsilon += (2*i + 1) * b_n[i] * sin_sin((2*i + 1), theta)
epsilon *= alpha_w/2

alpha_e = alpha_w - epsilon
fig, ax = plt.subplots()
ax.grid()
ax.plot(np.cos(theta), alpha_e, 'k')
ax.set_xlabel(r"$\xi$")
ax.set_ylabel(r"$\alpha_e$")
ax.set_title("Effective angle of attack")
fig.savefig("HW2_images/alpha_e_1.eps", format='eps')

# Induced drag
C_Di = 0
for i in range(num):
    C_Di += (2*i + 1) * b_n[i] * b_n[i]
C_Di *= np.pi/4 * A_r * alpha_w*alpha_w

print("CD_i =", C_Di)

e = 1/(np.pi * A_r) * C_L * C_L/C_Di
print("e =", e)

# Vortex
Gamma_0 = Gamma_norm[len(Gamma_norm)//2] * alpha_w
b_0 = (U_inf * U_inf)/(2 * Gamma_0 * U_inf * b)
s = b_0/b
r_c_norm = s * np.exp(-(4 * s*s/e + 1/2))
print("Gamma_0/U_inf b =", Gamma_0)
print("s =", s)
print("r_c/b =", r_c_norm)


# Induced velocity
xsi = np.linspace(-2, 2, num=201, endpoint=True)
w = np.zeros_like(xsi)
w += Gamma_0/(2*np.pi) * (xsi-s)/((xsi-s)*(xsi-s) + r_c_norm*r_c_norm)
w += -Gamma_0/(2*np.pi) * (xsi+s)/((xsi+s)*(xsi+s) + r_c_norm*r_c_norm)

fig, ax = plt.subplots()
ax.grid()
ax.plot(xsi, w, 'k')
ax.set_xlabel(r"$\xi$")
ax.set_ylabel(r"$\frac{w_v}{U_{\infty}}$")
ax.set_title("Velocity profile in the far wake")
fig.savefig("HW2_images/w_1.eps", format='eps')

#------2------#

print("------2------")

ds_norm = 0.5 * (1 + 1.5 * s)
dx_norm = ds_norm + s/2 + 1/2
print("ds_norm = 0.5 * (1 + 1.5 * s) =", ds_norm)
theta = np.linspace(0, np.pi, num=101, endpoint=True)
alpha_v = np.zeros_like(theta)
alpha_v += Gamma_0/(2*np.pi) * (np.cos(theta) + dx_norm-s)/((np.cos(theta) + dx_norm-s)*(np.cos(theta) + dx_norm-s) + r_c_norm*r_c_norm)
alpha_v += -Gamma_0/(2*np.pi) * (np.cos(theta) + dx_norm+s)/((np.cos(theta) + dx_norm+s)*(np.cos(theta) + dx_norm+s) + r_c_norm*r_c_norm)

plt.figure()
plt.plot(np.cos(theta), alpha_v)


A = np.zeros((2 * num, 2 * num))
c = np.zeros(2 * num)

for i in range(1, 2 * num + 1):
    c[i-1] = np.trapz(r(theta) * q(theta, i) * alpha_v, theta)
    for j in range(1, 2 * num + 1):
        A[i-1, j-1] = np.trapz(q(theta, j) * q(theta, i), theta)

C_n = solve(A, c)


# Angle of attack
alpha = 1/b_n[0] * (2 * C_L/(np.pi * A_r) - C_n[0])
print("alpha =", np.degrees(alpha), "°")

# Span loading
B_n = np.zeros_like(C_n)
for i in range(num):
    B_n[2 * i] += alpha * b_n[i] + C_n[2 * i]
    B_n[2 * i + 1] += C_n[2 * i + 1]


Gamma_norm = np.zeros_like(theta)
for i in range(2 * num):
    Gamma_norm += B_n[i] * np.sin((i+1) * theta)

K_trim = 2 * A_r * Gamma_norm
fig, ax = plt.subplots()
ax.grid()
ax.plot(np.cos(theta), K_trim, "k")
ax.set_xlabel(r"$\xi$")
ax.set_ylabel("K")
ax.set_title("Span loading")
fig.savefig("HW2_images/Span_loading_2.eps", format='eps')


# AoA
epsilon = np.zeros_like(theta)
for i in range(2 * num):
    epsilon += (i+1) * B_n[i] * sin_sin((i+1), theta)
epsilon *= 0.5


alpha_e = alpha + alpha_v - epsilon

fig, ax = plt.subplots()
ax.grid()
ax.plot(np.cos(theta), alpha_e, 'k')
ax.set_xlabel(r"$\xi$")
ax.set_ylabel(r"$\alpha_e$")
ax.set_title("Effective angle of attack")
fig.savefig("HW2_images/alpha_e_2.eps", format='eps')


# CDi
C_Di = A_r * np.trapz(Gamma_norm * (epsilon-alpha_v) * np.sin(theta), theta)

print("CD_i =", C_Di)

# CM
C_M = A_r/2 * np.trapz(Gamma_norm * np.cos(theta) * np.sin(theta), theta)
print("CM =", C_M)


#------3------#

f = np.zeros_like(theta)
f -= reg_step(theta, np.arccos(0.9), np.arccos(0.7))
f += reg_step(theta, np.arccos(-0.7), np.arccos(-0.9))
f *= tau

A = np.zeros((num, num))
c = np.zeros(num)
for i in range(2, 2 * num + 1, 2):
    c[i//2 - 1] = np.trapz(r(theta) * q(theta, i) * f, theta)
    for j in range(2, 2 * num + 1, 2):
        A[i//2 - 1, j//2 - 1] = np.trapz(q(theta, j) * q(theta, i), theta)

d_n = solve(A, c)

Gamma_norm = np.zeros_like(theta)
for i in range(num):
    Gamma_norm += d_n[i] * np.sin(2*(i + 1) * theta)

# delta a
delta_a = -2 * C_M/(A_r * np.trapz(Gamma_norm * np.cos(theta) * np.sin(theta), theta))
print("------3------")
print("delta_a =", np.degrees(delta_a))

# Span loading
D_n = np.zeros_like(C_n)
for i in range(num):
    D_n[2 * i] += alpha * b_n[i] + C_n[2 * i]
    D_n[2 * i + 1] += delta_a * d_n[i] + C_n[2 * i + 1]

Gamma_norm_tot = np.zeros_like(theta)
for i in range(2 * num):
    Gamma_norm_tot += D_n[i] * np.sin((i+1) * theta)

K = 2 * A_r * Gamma_norm_tot
fig, ax = plt.subplots()
ax.grid()
ax.plot(np.cos(theta), K, "k", label='Trimmed case')
ax.plot(np.cos(theta), K_trim, "k--", label='Untrimmed case')
ax.set_xlabel(r"$\xi$")
ax.set_ylabel("K")
ax.set_title("Span loading")
ax.legend(framealpha=1)
fig.savefig("HW2_images/Span_loading_3.eps", format='eps')


# C_Di
epsilon = np.zeros_like(theta)
for i in range(2 * num):
    epsilon += (i+1) * D_n[i] * sin_sin((i+1), theta)
epsilon *= 0.5

C_Di = A_r * np.trapz(Gamma_norm_tot * (epsilon - alpha_v + f * delta_a) * np.sin(theta), theta)
print("CD_i =", C_Di)

plt.show()
