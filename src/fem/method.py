import numpy as np
import scipy.sparse as sp
import matplotlib.tri as tri

def MassMatrix(X, Y, N, triangles):
    """
    Matriz de Masa, M.
    """

    M = sp.lil_matrix((N, N))

    aux = np.array([
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2]
    ])

    for triangle in triangles:

        n0, n1, n2 = triangle

        x0, y0 = X[n0], Y[n0]
        x1, y1 = X[n1], Y[n1]
        x2, y2 = X[n2], Y[n2]

        A = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2

        Mloc = (A / 12) * aux

        nodes = [n0, n1, n2]
        for i in range(3):
            for j in range(3):
                M[nodes[i], nodes[j]] += Mloc[i, j]

    M = M.tocsc()

    return M


def StiffnessMatrix(X, Y, N, triangles):
    """
    Matriz de Rigidez, K.
    """

    K = sp.lil_matrix((N, N))

    for triangle in triangles:

        n0, n1, n2 = triangle

        x0, y0 = X[n0], Y[n0]
        x1, y1 = X[n1], Y[n1]
        x2, y2 = X[n2], Y[n2]

        A = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2

        b = np.array([
            y1 - y2,
            y2 - y0,
            y0 - y1
        ])

        c = np.array([
            x2 - x1,
            x0 - x2,
            x1 - x0
        ])

        Kloc = (np.outer(b, b) + np.outer(c, c)) / (4 * A)

        nodes = [n0, n1, n2]
        for i in range(3):
            for j in range(3):
                K[nodes[i], nodes[j]] += Kloc[i, j]

    K = K.tocsc()

    return K


def LoadVector(X, Y, N, triangles, f, t):
    """
    Vector de Carga, F(t).
    """

    F = np.zeros(N)

    for triangle in triangles:

        n0, n1, n2 = triangle

        x0, y0 = X[n0], Y[n0]
        x1, y1 = X[n1], Y[n1]
        x2, y2 = X[n2], Y[n2]

        A = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2

        xc = (x0 + x1 + x2) / 3
        yc = (y0 + y1 + y2) / 3

        val = (A / 3) * f(xc, yc, t)

        F[n0] += val
        F[n1] += val
        F[n2] += val

    return F


def BTGalerkin(
    f, beta:float,
    Lx:float, Ly:float, T:float,
    u0,
    Ns:int, Nt:int
) -> tuple[np.ndarray, np.ndarray, tri.Triangulation, np.ndarray, np.ndarray]:

    """
    Resuelve la ecuación del calor en un dominio rectangular con condiciones de Neumann homogéneas 
    utilizando el método BT-Galerkin (Método de Elementos Finitos).
    """

    dt = T / Nt
    
    x = np.linspace(0, Lx, Ns+1)
    y = np.linspace(0, Ly, Ns+1)
    t = np.linspace(0, T, Nt+1)

    X, Y = np.meshgrid(x, y, indexing="ij")

    X = X.flatten()
    Y = Y.flatten()

    N = X.shape[0]

    TRI = tri.Triangulation(X, Y)

    M = MassMatrix(X, Y, N, TRI.triangles)
    K = StiffnessMatrix(X, Y, N, TRI.triangles)

    U = np.zeros((N, Nt+1))
    U[:, 0] = u0(X, Y)
    solve = sp.linalg.factorized(M + dt * beta * K)
    for k in range(Nt):
        rhs = M @ U[:, k] + dt * LoadVector(X, Y, N, TRI.triangles, f, t[k+1])
        U[:, k+1] = solve(rhs)

    return X, Y, TRI, t, U
