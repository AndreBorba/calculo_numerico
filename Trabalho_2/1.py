import numpy as np
import scipy.sparse
import matplotlib.pylab as plt
from matplotlib import cm
import matplotlib.animation as animation

# Converte as coordenadas (i, j) da grade para o índice do vetor correspondente
def ij2n(i, j, N1):
    return i + j * N1

def BuildMatrizesEigen(N1, N2, sigma, rho, e, delta):
    nunk = N1 * N2

    # Matriz de rigidez K: Construí-la como uma matriz esparsa
    d1 = 4.0 * np.ones(nunk)
    d2 = -np.ones(nunk - 1)
    d3 = -np.ones(nunk - N1)
    K = (sigma / delta**2) * scipy.sparse.diags([d3, d2, d1, d2, d3], [-N1, -1, 0, 1, N1], format='csr')

    # Forçar os autovalores associados aos pontos de fronteira a serem um número grande
    big_number = 10000
    Iden = big_number * scipy.sparse.identity(nunk, format='csr')

    # Lados verticais
    for k in range(0, N2):
        Ic = ij2n(0, k, N1)  # Esquerda
        K[Ic, :], K[:, Ic] = Iden[Ic, :], Iden[:, Ic]

        Ic = ij2n(N1 - 1, k, N1)  # Direita
        K[Ic, :], K[:, Ic] = Iden[Ic, :], Iden[:, Ic]
        
    # Lados horizontais
    for k in range(0, N1):
        Ic = ij2n(k, 0, N1)  # Baixo
        K[Ic, :], K[:, Ic] = Iden[Ic, :], Iden[:, Ic]

        Ic = ij2n(k, N2 - 1, N1)  # Topo
        K[Ic, :], K[:, Ic] = Iden[Ic, :], Iden[:, Ic]

    # Matriz de massa: Caso simples, múltiplo da identidade
    M = rho * e * scipy.sparse.identity(nunk, format='csr')
    
    return K, M

# Função para calcular as frequências de oscilação
def calcula_frequencias(N1, N2, sigma, rho, e):
    delta = 1.0 / (N1 - 1)
    K, M = BuildMatrizesEigen(N1, N2, sigma, rho, e, delta)
    Lam, Q = scipy.sparse.linalg.eigsh(K, k=4, M=M, which='SM')
    omegas = np.sqrt(Lam)
    return omegas, Q

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
L1 = 1
L2 = 1
sigma = 1  # Tensão superficial
rho = 1  # Densidade
e = 1  # Espessura

# Dicionário para armazenar resultados
resultados = {}

for N in N_list:
    omegas, Q = calcula_frequencias(N, N, sigma, rho, e)  # N iguais --> N1 = N2 = N
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