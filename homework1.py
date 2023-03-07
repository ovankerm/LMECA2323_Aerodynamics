import numpy as np
from matplotlib import pyplot as plt

N = 65
x_c, dx = np.linspace(0, 1, endpoint=True, retstep=True, num=N)

h_c = 4 * 0.05 * x_c * (1 - x_c)

fig, ax = plt.subplots()
ax.plot(x_c, h_c, 'black')
ax.plot(x_c, -h_c, 'black')
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-1, 1)

plt.show()