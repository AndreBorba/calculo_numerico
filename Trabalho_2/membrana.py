import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ij2n(i, j, N):
    return i + j * N

def BuildMatrizesEigen(N1, N2, sigma, rho, e, delta):
    nunk = N1 * N2

    # Stiffness matrix K: Build it as a sparse matrix 
    d1 = 4.0 * np.ones(nunk)
    d2 = -np.ones(nunk - 1)
    d3 = -np.ones(nunk - N1)
    K = (sigma / delta ** 2) * scipy.sparse.diags([d3, d2, d1, d2, d3], [-N1, -1, 0, 1, N1], format='csr')

    # Force the eigenvalues associated to boundary points 
    # to be a big number as compared to fundamental modes
    big_number = 10000
    Iden = big_number * scipy.sparse.identity(nunk, format='csr')

    # Lados verticais
    for k in range(0, N2):
        Ic = ij2n(0, k, N1)  # Left
        K[Ic, :], K[:, Ic] = Iden[Ic, :], Iden[:, Ic]

        Ic = ij2n(N1 - 1, k, N1)  # Right
        K[Ic, :], K[:, Ic] = Iden[Ic, :], Iden[:, Ic]

    # Lados horizontais
    for k in range(0, N1):
        Ic = ij2n(k, 0, N1)  # Bottom
        K[Ic, :], K[:, Ic] = Iden[Ic, :], Iden[:, Ic]

        Ic = ij2n(k, N2 - 1, N1)  # Top
        K[Ic, :], K[:, Ic] = Iden[Ic, :], Iden[:, Ic]

    # Mass matrix: Simple case, multiple of identity
    M = rho * e * scipy.sparse.identity(nunk, format='csr')

    return K, M

def compute_frequencies(N1, N2, sigma, rho, e):
    delta = 1.0 / (N1 - 1)
    K, M = BuildMatrizesEigen(N1, N2, sigma, rho, e, delta)
    Lam, Q = scipy.sparse.linalg.eigsh(K, k=4, M=M, which='SM')
    omegas = np.sqrt(Lam)
    return omegas, Q

def PlotaMembrane(N1, N2, L1, L2, Wplot):
    x = np.linspace(0, L1, N1)
    y = np.linspace(0, L2, N2)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Wplot, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Amplitude')
    ax.set_title('Modo de Vibração da Membrana')
    plt.show()

# Parameters
sigma = 1.0
rho = 1.0
e = 1.0

# Grid sizes to consider
grid_sizes = [11, 21, 31, 41, 51, 61, 81, 101]

# Store results
results = {}

for N in grid_sizes:
    omegas, Q = compute_frequencies(N, N, sigma, rho, e)
    results[N] = (omegas, Q)

# Print results in table form
print(f"{'Grid Size':<10}{'Frequency 1':<15}{'Frequency 2':<15}{'Frequency 3':<15}{'Frequency 4':<15}")
for N, (omegas, _) in results.items():
    print(f"{N:<10}{omegas[0]:<15.8f}{omegas[1]:<15.8f}{omegas[2]:<15.8f}{omegas[3]:<15.8f}")

# Plot results
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.plot(grid_sizes, [results[N][0][i] for N in grid_sizes], label=f'Frequency {i+1}')
plt.xlabel('Grid Size (N)')
plt.ylabel('Frequency (rad/s)')
plt.title('Frequencies of Vibration as Function of Grid Size')
plt.legend()
plt.grid(True)
plt.show()

# Choose a grid size and mode to plot
N = 41
omegas, Q = results[N]
mode_number = 0  # Select the mode number to plot (0-based index)
mode = Q[:, mode_number]

# Reshape mode to 2D grid
Wplot = mode.reshape(N, N)
PlotaMembrane(N, N, 1.0, 1.0, Wplot)
