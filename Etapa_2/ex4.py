import numpy as np

def Assembly(conec, C):
    
    lista_mtx = []  # lista que conterá os valores da matriz
    for i in conec: # loops para formar a lista aicma
        for j in i:
            lista_mtx.append(j)

    nv = max(lista_mtx) #numero de nos
    nc = len(conec) #numero de canos
    A = np.zeros(shape=(nv,nv))
    
    for k in range(nc):
        n1 = conec[k,0]
        n2 = conec[k,1]

        A[n1-1][n1-1] += C[k]
        A[n2-1][n2-1] += C[k]
        A[n1-1][n2-1] -= C[k]
        A[n2-1][n1-1] -= C[k]


    return A

def SolveNetwork(conec, C, natm, nB, QB):
    Atilde = Assembly(conec, C)
    for i in range(len(Atilde)):
        Atilde[natm-1, i] = 0
    Atilde[natm-1, natm-1] = 1

    b = np.zeros(len(Atilde))
    for i in range(len(b)):
        if(i == nB-1):
            b[i] = QB
        else:
            b[i] = 0

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
        for j in range(nv):
            if(j == conec[k,0]):
                D[k,j] = 1
            elif(j == conec[k,1]):
                D[k,j] = -1
            else:
                D[k,j] = 0

    return D

conec = np.array([
    [1,2],
    [2,3],
    [3,4],
    [4,5],
    [5,2],
    [5,3],
    [5,1]
])

C = np.array([2,2,1,2,1,2,2])

pressure = SolveNetwork(conec, C, 3, 1, 3)
K = escreve_mtx_diagonal(C)
D = escreve_mtx_D(C, conec)

transposta_pressure = pressure.transpose()  # calculando matrizes transpostas para determinar a potencia consumida pela bomba
transposta_D = D.transpose()

potencia = transposta_pressure@(transposta_D@K@D)@pressure
print(potencia)

