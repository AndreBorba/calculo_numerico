import numpy as np
import scipy.sparse
import matplotlib.pylab as plt
from matplotlib import cm
import matplotlib.animation as animation

def ij2n(i, j, N1):
    return i + j * N1

def ponto_no_triangulo(pt, v1, v2, v3):
    # Função para verificar se um ponto está dentro de um triângulo
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign(pt, v1, v2) < 0.0
    b2 = sign(pt, v2, v3) < 0.0
    b3 = sign(pt, v3, v1) < 0.0

    return ((b1 == b2) and (b2 == b3))

def BuildMatrizesEigenTriangular(N1, N2, sigma, rho, e, delta, p1, p2, p3):
    nunk = N1 * N2  
    
    d1 = 4.0 * np.ones(nunk)
    d2 = -np.ones(nunk - 1)
    d3 = -np.ones(nunk - N1)
    K = (sigma / delta**2) * scipy.sparse.diags([d3, d2, d1, d2, d3], [-N1, -1, 0, 1, N1],  format='csr')                                               
    
    big_number = 100000
    Iden = big_number * scipy.sparse.identity(nunk, format='csr')

    mask = np.zeros((N1, N2), dtype=bool)
    for i in range(N1):
        for j in range(N2):
            x = i * delta
            y = j * delta
            if ponto_no_triangulo((x, y), p1, p2, p3):
                mask[i, j] = True

    for i in range(N1):
        for j in range(N2):
            if not mask[i, j]:
                Ic = ij2n(i, j, N1)
                K[Ic, :] = Iden[Ic, :]
                K[:, Ic] = Iden[:, Ic]

    M = rho * e * scipy.sparse.identity(nunk, format='csr')
    
    return K, M

# Função para calcular as frequências de oscilação
def calcula_frequencias(N1, N2, sigma, rho, e, p1, p2, p3):
    delta = 1.0 / (N1 - 1)
    K, M = BuildMatrizesEigenTriangular(N1, N2, sigma, rho, e, delta, p1, p2, p3)
    Lam, Q = scipy.sparse.linalg.eigsh(K, k=4, M=M, which='SM')
    omegas = np.sqrt(Lam)
    return omegas, Q

# Verificação de que os pontos formam um triângulo
def is_valid_triangle(p1, p2, p3):
    return not np.isclose((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])), 0)

# Função para plotar as curvas de nível constante de temperatura
def PlotaMembrane(ax, N1, N2, L1, L2, W, N, k):
    x = np.linspace(0, L1, N1)
    y = np.linspace(0, L2, N2)
    X, Y = np.meshgrid(x, y)
    Z = np.copy(W)
    ax.set_aspect('equal')
    ax.set(xlabel='x', ylabel='y', title=f'Deslocamento vertical, N={N}\n frequência de oscilação: {k+1}')
    im = ax.contourf(X, Y, Z, 20)
    ax.contour(X, Y, Z, 20, linewidths=0.25, colors='k')
    plt.colorbar(im, ax=ax)

def PlotaSurface(ax, N1, N2, L1, L2, W, N, k):
    x = np.linspace(0, L1, N1)
    y = np.linspace(0, L2, N2)
    X, Y = np.meshgrid(x, y)
    Z = np.copy(W)
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set(xlabel='x', ylabel='y', title=f'Deslocamento vertical, N={N}\n frequência de oscilação: {k+1}')
    ax.set_zlim(-1.01, 1.01)

# Parâmetros
N_list = [11, 21, 31, 41, 51, 61, 81, 101]  # Lista de tamanhos de grade (N1 = N2 = N)
sigma = 1
rho = 1
e = 1
L1 = 1.0
L2 = 1.0


# Definindo os pontos do triângulo
p1 = (0.2, 0.2)
p2 = (0.8, 0.2)
p3 = (0.5, 0.8)

# Dicionário para armazenar resultados
resultados = {}

if is_valid_triangle(p1, p2, p3):
    for N in N_list:
        omegas, Q = calcula_frequencias(N, N, sigma, rho, e, p1, p2, p3)  # N iguais --> N1 = N2 = N
        resultados[N] = (omegas, Q)

    # Mostrando os resultados em forma de tabela
    print(f"{'Grid Size':<10}{'Frequency 1':<15}{'Frequency 2':<15}{'Frequency 3':<15}{'Frequency 4':<15}")
    for N, (omegas, _) in resultados.items():
        print(f"{N:<10}{omegas[0]:<15.8f}{omegas[1]:<15.8f}{omegas[2]:<15.8f}{omegas[3]:<15.8f}")

    # Plotando modos de oscilação
    for N in N_list:
        omegas, Q = resultados[N]
        for k in range(4):
            mode = Q[:, k]
            Wplot = mode.reshape(N, N)
            
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
            PlotaMembrane(axs[0], N, N, L1, L2, Wplot, N, k)
            ax = fig.add_subplot(122, projection='3d')
            PlotaSurface(ax, N, N, L1, L2, Wplot, N, k)
            
            plt.show()
else:
    print("Os pontos fornecidos não formam um triângulo válido.")


# Make a movie (it is a bit slow)
# Adicionando animação para N = 11

N = 11
omegas, Q = resultados[N]
Wplot = Q[:, 0].reshape(N, N)

x = np.linspace(0, L1, N)
y = np.linspace(0, L2, N)
X, Y = np.meshgrid(x, y)
Z = np.copy(Wplot)

fps = 4 # frame per sec
frn = 20 # frame number of the animation
zarray = np.zeros((N, N, frn))
for i in range(frn):
    zarray[:,:,i] = 10 * np.sin(2 * np.pi * i / frn) * Wplot  # Multiplico x10 para exagerar o deslocamento

def update_plot(frame_number, zarray, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(X, Y, zarray[:,:,frame_number], cmap=cm.coolwarm, linewidth=0, antialiased=False)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set(xlabel='x', ylabel='y', title='Deslocamento vertical')
plot = [ax.plot_surface(X, Y, zarray[:,:,0], color='0.75', rstride=1, cstride=1)]
ax.set_zlim(-10, 10)

ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(zarray, plot), interval=1000/fps)
plt.show()

fn = 'membanim'
ani.save(fn + '.gif', writer='imagemagick', fps=fps)
print('\n ... Movie generated\n')