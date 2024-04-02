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

A = Assembly(conec, C)
print(A)
