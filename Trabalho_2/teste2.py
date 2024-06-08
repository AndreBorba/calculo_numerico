import numpy as np
import scipy.sparse
import matplotlib.pylab as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Converte as coordenadas (i, j) da grade para o índice do vetor correspondente
def ij2n(i, j, N1):
    return i + j * N1

def BuildMatrizesEigenTriangular(N1, N2, sigma, rho, e, delta, L):
    nunk = N1 * N2  
    
    # Criando a matriz de rigidez K como uma matriz esparsa
    d1 = 4.0 * np.ones(nunk)  # Diagonal principal com valor 4.0
    d2 = -np.ones(nunk - 1)   # Diagonais secundárias com valor -1.0
    d3 = -np.ones(nunk - N1)  # Diagonais terciárias com valor -1.0
    K = (sigma / delta**2) * scipy.sparse.diags([d3, d2, d1, d2, d3], [-N1, -1, 0, 1, N1], format='csr')
                                               
    # Força os autovalores associados aos pontos de fronteira a serem grandes
    big_number = 10000
    Iden = big_number * scipy.sparse.identity(nunk, format='csr')

    # Máscara para identificar os pontos dentro da membrana triangular
    mask = np.zeros((N1, N2), dtype=bool)
    for i in range(N1):
        for j in range(N2):
            x = i * delta  # Coordenada x do ponto (i, j) na grade
            y = j * delta  # Coordenada y do ponto (i, j) na grade
            if x >= 0 and y >= 0 and (y <= (-np.sqrt(3) * x + np.sqrt(3) * L)):
                mask[i, j] = True

    # Aplicando a máscara na matriz de rigidez K
    for i in range(N1):
        for j in range(N2):
            if not mask[i, j]:  # Se o ponto está fora da membrana triangular
                Ic = ij2n(i, j, N1)
                K[Ic, :] = Iden[Ic, :]  # Define as linhas e colunas correspondentes em K
                K[:, Ic] = Iden[:, Ic]

    # Matriz de massa: Caso simples, múltiplo da identidade
    M = rho * e * scipy.sparse.identity(nunk, format='csr')
    
    return K, M

# Função para calcular as frequências de oscilação
def calcula_frequencias_triangular(N1, N2, sigma, rho, e, L):
    delta = 1.0 / (N1 - 1)
    K, M = BuildMatrizesEigenTriangular(N1, N2, sigma, rho, e, delta, L)
    Lam, Q = scipy.sparse.linalg.eigsh(K, k=4, M=M, which='SM')
    omegas = np.sqrt(Lam)
    return omegas, Q

# Parâmetros
N_list = [11, 21, 31, 41, 51, 61, 81, 101]  # Lista de tamanhos de grade (N1 = N2 = N)
L = 1  # Comprimento do lado do triângulo
sigma = 1  # Tensão superficial
rho = 1  # Densidade
e = 1  # Espessura

# Dicionário para armazenar resultados
resultados_triangular = {}

for N in N_list:
    omegas, Q = calcula_frequencias_triangular(N, N, sigma, rho, e, L)  # N iguais --> N1 = N2 = N
    resultados_triangular[N] = (omegas, Q)

# Mostrando os resultados em forma de tabela
print(f"{'Grid Size':<10}{'Frequency 1':<15}{'Frequency 2':<15}{'Frequency 3':<15}{'Frequency 4':<15}")
for N, (omegas, _) in resultados_triangular.items():
    print(f"{N:<10}{omegas[0]:<15.8f}{omegas[1]:<15.8f}{omegas[2]:<15.8f}{omegas[3]:<15.8f}")

# Plotando modos de oscilação
for N in N_list:
    omegas, Q = resultados_triangular[N]
    for k in range(4):
        mode = Q[:, k]
        Wplot = mode.reshape(N, N)
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        PlotaMembrane(axs[0], N, N, L, L, Wplot, N, k)
        ax = fig.add_subplot(122, projection='3d')
        PlotaSurface(ax, N, N, L, L, Wplot, N, k)
        
        plt.show()

# Make a movie (it is a bit slow)
# Adicionando animação para N = 11

N = 11
omegas, Q = resultados_triangular[N]
Wplot = Q[:, 0].reshape(N, N)

x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
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
plot = [ax.plot_surface(X, Y, zarray[:,:,0], color='0.75', rstride=
