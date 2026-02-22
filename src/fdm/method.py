import numpy as np
import scipy.sparse as sp

def BTCS(
    f, beta:float,
    Lx:float, Ly:float, T:float,
    u0,
    Nx:int, Ny:int, Nt:int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Resuelve la ecuación del calor en un dominio rectangular con condiciones de Neumann homogéneas 
    utilizando el método BT-CS (Método de Diferencias Finitas).
    """

    hx = Lx / Nx
    hy = Ly / Ny
    dt = T / Nt

    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    t = np.linspace(0, T, Nt+1)

    lambda_x = beta * dt / hx**2
    lambda_y = beta * dt / hy**2

    N = (Nx + 1) * (Ny + 1)

    A = sp.lil_matrix((N, N))

    for i in range(Nx+1):
        for j in range(Ny+1):

            l = i + (Nx + 1) * j
            
            A[l, l] = 1 + 2 * (lambda_x + lambda_y)

            if i == 0:
                A[l, l+1] = -2 * lambda_x
            elif i == Nx:
                A[l, l-1] = -2 * lambda_x
            else: 
                A[l, l+1] = -lambda_x
                A[l, l-1] = -lambda_x

            if j == 0:
                A[l, l+(Nx+1)] = -2 * lambda_y
            elif j == Ny:
                A[l, l-(Nx+1)] = -2 * lambda_y
            else:
                A[l, l+(Nx+1)] = -lambda_y
                A[l, l-(Nx+1)] = -lambda_y

    A = A.tocsc()

    X, Y = np.meshgrid(x, y, indexing="ij")

    U = np.zeros((N, Nt+1))
    U[:, 0] = u0(X, Y).reshape(-1, order="F")
    solve = sp.linalg.factorized(A)
    for k in range(Nt):
        rhs = U[:, k] + dt * f(X, Y, t[k+1]).reshape(-1, order="F")
        U[:, k+1] = solve(rhs)

    return x, y, t, U
