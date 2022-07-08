import matplotlib.pyplot as plt
import numpy as np

mu0 = 4e-7 * np.pi  # H m-1
I = 1.5  # A
l = 0.005  # m
R0 = 0.001  # m

F = (mu0 / (2 * np.pi)) * I ** 2 * l / R0

print(F)

dim = 0.003  # m
N = 1000

x = np.outer(np.ones(N), np.linspace(-dim, dim, N))
z = np.outer(np.linspace(-2 * dim, 0, N), np.ones(N))

B = np.zeros_like(x)
for i ,pos in enumerate([-R0 / 2, R0 / 2]):
    r = np.sqrt((x - pos) ** 2 + (z + 0.001) ** 2)
    r = np.clip(r, 0.00025, np.inf)
    if i == 0 :
        B = (mu0 / (2 * np.pi)) * I / r
    else:
        B += (mu0 / (2 * np.pi)) * I / r


# https://www.supermagnete.de/eng/faq/How-do-you-calculate-the-magnetic-flux-density#formula-for-block-magnet-flux-density




grad = np.diff(B, axis=0, prepend=np.nan)


# plt.imshow(np.log(B), origin='lower')
plt.contour(x*1000, z*1000, B, levels=30, cmap="Reds")
# plt.imshow(np.log(B), cmap = 'Reds', origin='lower')
# plt.imshow(grad, cmap = 'Reds', origin='lower')
plt.colorbar()
plt.show()
