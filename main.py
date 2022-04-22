import efit
import random
import math

cnt = 0
eps = 0.1
X = []
Y = []
Z = []

# imposed center and radii
x0, y0, z0 = 1, 2, 3
a, b, c = 4, 5, 6

while cnt < 10000:
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

# recovered center and radii
print(*center)
print(*radii)
