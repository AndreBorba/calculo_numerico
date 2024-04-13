import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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

    Atilde_sparse = sp.csr_matrix(Atilde)
    pressure = spla.spsolve(Atilde_sparse, b)

    return pressure

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

def RandomFailFinos(C, p0, Centup, Cfino):
  Cnew = np.copy(C)
  nc = len(C)
  for k in range(nc):
    x = np.random.rand()
    if(x <= p0 and C[k] == Cfino):
      Cnew[k] = Centup

  return Cnew

def MonteCarlo(C, p0, Centup, Cfino, conec, natm, nB, QB, num_simulacoes):
    pressao_acima_12 = 0
    for _ in range(num_simulacoes):
        Cnew = RandomFailFinos(C, p0, Centup, Cfino)
        pressoes = SolveNetwork(conec, Cnew, natm, nB, QB)
        if np.max(pressoes) >= 12:
            pressao_acima_12 += 1
    
    return pressao_acima_12 / num_simulacoes
  
CH = 2.0
CV = 2.0
nx, ny = 10, 10
nv, nc, conec, C, coord = GeraRede(nx, ny , CH, CV)
lst = [0, 1, 2, 30, 31, 32, 93, 103, 113]
C[lst] = 20.0


Centup = 0.2
Cfino = 2
natm = nv - 1
nB = 0
QB = 3
num_simulacoes = 6500
# probabilidade = MonteCarlo(C, 0.4, Centup, Cfino, conec, natm, nB, QB, num_simulacoes)
# print(probabilidade)

probs_p0 = np.linspace(0, 1, num=20)
probabilidades_MC = []
for p0 in probs_p0:
    probabilidade = MonteCarlo(C, p0, Centup, Cfino, conec, natm, nB, QB, num_simulacoes)
    probabilidades_MC.append(probabilidade)

plt.plot(probs_p0, probabilidades_MC, marker='o')
plt.xlabel('Probabilidade de Entupimento (p0)')
plt.ylabel('Probabilidade de Pressão Acima de 12 em algum nó')
plt.title("Probabilidade de pressão acima de 12 quando se varia o valor p0 (probabilidade de entupimento de canos finos)")
plt.grid()
plt.show()


num_realizacoes = np.linspace(1, num_simulacoes, num=50, dtype=int)
valores_p0 = [0.2, 0.4, 0.6, 0.8]
plt.figure(figsize=(10,6))

for p0 in valores_p0:
  probabilidades_MC.clear()

  for realizacoes in num_realizacoes:
    realizacoes = int(realizacoes)
    probabilidade = MonteCarlo(C, p0, Centup, Cfino, conec, natm, nB, QB, realizacoes)
    probabilidades_MC.append(probabilidade)
  plt.plot(num_realizacoes, probabilidades_MC, label=f'p0 = {p0}')

plt.xlabel("Número de realizações")
plt.ylabel("Probababilidade de pressão acima de 12")
plt.legend()
plt.title("Probabilidade de pressão acima de 12 para diferentes valores de simulações")
plt.grid()
plt.show()