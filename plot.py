import efit
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

cnt = 0
eps = 0.1
X = []
Y = []
Z = []

# imposed center and radii
x0, y0, z0 = 1, 2, 3
a, b, c = 4, 5, 6

while cnt < 300:
    x = random.uniform(-10, 10)
    y = random.uniform(-10, 10)
    z = random.uniform(-10, 10)
    if abs((x - x0)**2 / a**2 + (y - y0)**2 / b**2 +
           (z - z0)**2 / c**2 - 1) < eps:
        cnt += 1
        X.append(x)
        Y.append(y)
        Z.append(z)

center, radii, evecs, v, chi2 = efit.efit(X, Y, Z)

u = np.linspace(0, 2 * math.pi, 100)
v = np.linspace(0, math.pi, 100)

x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))
x, y, z = np.einsum('ij,jkl,j', evecs, (x, y, z), radii)
x += center[0]
y += center[1]
z += center[2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
ax.plot_surface(x, y, z)
ax.scatter(X, Y, Z)
plt.savefig("efit.svg")
sys.stderr.write("efit.svg\n")
