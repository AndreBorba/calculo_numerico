import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # precisa desse import pra rodar no meu pc


def Assembly(conec, C):
    
    lista_mtx = []  # lista que conterá os valores da matriz
    for i in conec: # loops para formar a lista aicma
        for j in i:
            lista_mtx.append(j)

    nv = max(lista_mtx) + 1 #numero de nos. Neste exercicio acrescentei o +1 pois osnós começam a ser enumerados em 0
    nc = len(conec) #numero de canos
    A = np.zeros(shape=(nv,nv))
    
    for k in range(nc):
        n1 = conec[k,0]
        n2 = conec[k,1]

        A[n1][n1] += C[k]
        A[n2][n2] += C[k]
        A[n1][n2] -= C[k]
        A[n2][n1] -= C[k]


    return A

def SolveNetwork(conec, C, natm, nB, QB):
    Atilde = Assembly(conec, C)
    for i in range(len(Atilde)):
        Atilde[natm, i] = 0
    Atilde[natm, natm] = 1

    b = np.zeros(len(Atilde))
    for i in range(len(b)):
        if(i == nB):
            b[i] = QB

    pressure = np.linalg.solve(Atilde, b)

    return pressure

def escreve_mtx_diagonal(C):
    nc = len(C) #numero de canos
    K = np.zeros(shape=(nc, nc))
    for i in range(nc):
        K[i,i] = C[i]
    return K

def escreve_mtx_D(C, conec):
    lista_mtx = []  # lista que conterá os valores da matriz de conectores
    for i in conec: # loops para formar a lista aicma
        for j in i:
            lista_mtx.append(j)
    
    nv = max(lista_mtx) # numero de nos
    nc = len(C) # numero de canos
    D = np.zeros(shape=(nc,nv))

    for k in range(nc):
        D[k,conec[k,0]] = 1
        D[k,conec[k,1]] = -1

    return D

def GeraRede(nx,ny,CH,CV):
    nv = nx*ny
    nc = (nx-1)*ny + (ny-1)*nx
    
    coord = np.zeros(shape=(nv,2))

    for i in range(nx):
      for j in range(ny):
        ig = i + j*nx
        coord[ig,0] = i
        coord[ig,1] = j

    conec = np.zeros(shape=(nc,2), dtype=int)
    C = np.zeros(nc)
    
    # Loop sobre canos horizontais
    for j in range(ny):
        for i in range(nx-1):
          k = j*(nx-1) + i
          conec[k,0] = j*nx + i
          conec[k,1] = j*nx + i+1
          C[k] = CH

    # Loop sobre canos verticais
    for i in range(nx):
      for j in range(ny-1):
          k = (nx-1)*ny + j*nx + i
          conec[k,0] = i + j*nx
          conec[k,1] = i + (j+1)*nx
          C[k] = CV

    return nv, nc, conec, C, coord

def PlotPressure(nx, ny, pressure):
    x = np.arange(0, nx, 1)
    y = np.arange(0, ny, 1)
    X,Y=np.meshgrid(x,y)
    Z = np.copy(pressure)
    Z.shape = (ny,nx)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal')
    ax.set(xlabel='x', ylabel='y', title='Contours of pressure')
    ax.grid()
    ax.set_xticks(x)
    ax.set_yticks(y)
    cp = plt.contourf(X, Y, Z, cmap='jet')
    cb = plt.colorbar(cp)
    plt.show()

def PlotPressureAsSurface(nx, ny, pressure):
    x = np.arange(0, nx, 1)
    y = np.arange(0, ny, 1)
    X,Y=np.meshgrid(x,y)
    Z = np.copy(pressure)
    Z.shape = (ny,nx)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal')
    ax.set(xlabel='x', ylabel='y', title='Contours of pressure')
    ax.grid()
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap='jet', edgecolor=None)
    ax.set(xlabel='$x$', ylabel='$y$', zlabel='$z$')
    ax.view_init(elev=30., azim=45)
    plt.show()

CH = 1.3
CV = 3.6
nx, ny = 10, 12

# Gerando a rede
nv, nc, conec, C, coord = GeraRede(nx,ny,CH,CV)

# determinando o numero do nó que está no canto superior direito (que é natm neste exemplo)
# os nós sao enumerados da esqurda para a direita e de baixo para cima, assim:
#                                                                              3 4 5                                     
#                                                                              0 1 2 
natm = nx*ny - 1

# canto inferior esquerdo (nB neste exemplo)
nB = 0
QB = 3

# Calculando o vetor de pressões 
pressure1 = SolveNetwork(conec, C, natm, nB, QB)
pressure2 = SolveNetwork(conec, C, 55, 0, QB)
pressure3 = SolveNetwork(conec, C, 6, 0, QB)

PlotPressure(nx, ny, pressure1)
PlotPressureAsSurface(nx, ny, pressure1)

PlotPressure(nx, ny, pressure2)
PlotPressureAsSurface(nx, ny, pressure2)

PlotPressure(nx, ny, pressure3)
PlotPressureAsSurface(nx, ny, pressure3)
