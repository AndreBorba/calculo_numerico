import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # precisa desse import pra rodar no meu pc
import time

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
nx_1, ny_1 = 10, 10
nv_1 = nx_1*ny_1

nx_2, ny_2 = 20, 20
nv_2 = nx_2*ny_2

nx_3, ny_3 = 40, 40
nv_3 = nx_3*ny_3

nx_4, ny_4 = 50, 50
nv_4 = nx_4*ny_4

nx_5, ny_5 = 60, 60
nv_5 = nx_5*ny_5

nx_6, ny_6 = 70, 70
nv_6 = nx_6*ny_6

nx_7, ny_7 = 80, 80
nv_7 = nx_7*ny_7

# Gerando a rede
nv_1, nc_1, conec_1, C_1, coord_1 = GeraRede(nx_1,ny_1,CH,CV)
nv_2, nc_2, conec_2, C_2, coord_2 = GeraRede(nx_2,ny_2,CH,CV)
nv_3, nc_3, conec_3, C_3, coord_3 = GeraRede(nx_3,ny_3,CH,CV)
nv_4, nc_4, conec_4, C_4, coord_4 = GeraRede(nx_4,ny_4,CH,CV)
nv_5, nc_5, conec_5, C_5, coord_5 = GeraRede(nx_5,ny_5,CH,CV)
nv_6, nc_6, conec_6, C_6, coord_6 = GeraRede(nx_6,ny_6,CH,CV)
nv_7, nc_7, conec_7, C_7, coord_7 = GeraRede(nx_7,ny_7,CH,CV)

# determinando o numero do nó que está no canto superior direito (que é natm neste exemplo)
# os nós sao enumerados da esqurda para a direita e de baixo para cima, assim:
#                                                                              3 4 5                                     
#                                                                              0 1 2 
natm_1 = nx_1*ny_1 - 1
natm_2 = nx_2*ny_2 - 1
natm_3 = nx_2*ny_3 - 1
natm_4 = nx_2*ny_4 - 1
natm_5 = nx_2*ny_5 - 1
natm_6 = nx_2*ny_6 - 1
natm_7 = nx_2*ny_7 - 1

# Calculando o vetor de pressões e os respectivos tempos de cálculo
t0 = time.time()
pressure1 = SolveNetwork(conec_1, C_1, natm_1, 0, 3)
t1 = time.time() - t0

t0 = time.time()
pressure2 = SolveNetwork(conec_2, C_2, natm_2, 0, 3)
t2 = time.time() - t0

t0 = time.time()
pressure3 = SolveNetwork(conec_3, C_3, natm_3, 0, 3)
t3 = time.time() - t0

t0 = time.time()
pressure4 = SolveNetwork(conec_4, C_4, natm_4, 0, 3)
t4 = time.time() - t0

t0 = time.time()
pressure5 = SolveNetwork(conec_5, C_5, natm_5, 0, 3)
t5 = time.time() - t0

t0 = time.time()
pressure6 = SolveNetwork(conec_6, C_6, natm_6, 0, 3)
t6 = time.time() - t0

t0 = time.time()
pressure7 = SolveNetwork(conec_7, C_7, natm_7, 0, 3)
t7 = time.time() - t0

'''
print('Tempo 1: ', t1)
print('Tempo 2: ', t2)
print('Tempo 3: ', t3)
print('Tempo 4: ', t4)
'''

# lista dos tempos e lista dos produtos nx*ny = nv 
tempos = [t1, t2, t3, t4, t5, t6, t7]
nv = [nv_1, nv_2, nv_3, nv_4, nv_5, nv_6, nv_7]

# plot dos graficos
plt.figure(figsize=(8,6))
plt.loglog(nv, tempos, marker = 'o', linestyle='-')
plt.xlabel('Número total de nós (nv)')
plt.ylabel('Tempo para calcular pressões (s)')
plt.title('Tempo de cálculo pressões versus número de nós na rede')
plt.grid()
plt.show()




