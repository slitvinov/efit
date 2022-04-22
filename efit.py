import numpy as np


def efit(x, y, z):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    D = [
        x * x + y * y - 2 * z * z, x * x + z * z - 2 * y * y, 2 * x * y,
        2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z, 1 + 0 * x
    ]
    D = np.array(D)

    d2 = x * x + y * y + z * z
    d2 = d2.reshape((d2.shape[0], 1))
    Q = np.dot(D, D.T)
    b = np.dot(D, d2)
    u = np.linalg.solve(Q, b)

    v = np.zeros((u.shape[0] + 1, u.shape[1]))
    v[0] = u[0] + u[1] - 1
    v[1] = u[0] - 2 * u[1] - 1
    v[2] = u[1] - 2 * u[0] - 1
    v[3:10] = u[2:9]

    A = np.array([
        v[0], v[3], v[4], v[6], v[3], v[1], v[5], v[7], v[4], v[5], v[2], v[8],
        v[6], v[7], v[8], v[9]
    ]).reshape((4, 4))

    center = np.linalg.solve(-A[:3, :3], v[6:9])
    T = np.eye(4)
    T[3, :3] = center.T
    center = center.reshape((3, ))
    R = T.dot(A).dot(T.conj().T)
    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])

    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    sgns = np.sign(evals)
    radii = np.sqrt(sgns / evals)

    d = np.array([x - center[0], y - center[1], z - center[2]])
    d = np.dot(d.T, evecs)
    d = np.array([d[:, 0] / radii[0], d[:, 1] / radii[1],
                  d[:, 2] / radii[2]]).T
    chi2 = np.sum(
        np.abs(1 - np.sum(d**2 * np.tile(sgns, (d.shape[0], 1)), axis=1)))

    return center, radii, evecs, v, chi2
