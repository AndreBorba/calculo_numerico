import numpy as np 
import scipy 


def algoritmo_francis(n, max_iter, tol):
    k = 0 # representa as iterações
    A = constroi_matriz(n) #matriz que conterá autovalores em sua diagonal principal
    V = np.identity(n) #matriz que conterá autovetores em suas colunas
    error = np.max(np.abs(A))
    
    while(error > tol and k < max_iter):
        Q, R = scipy.linalg.qr(A)
        A = R@Q
        V = V@Q
        error = np.max(np.abs(A)) #avaliar os valores da matriz inteira ou apenas um deles, como o A[n-1,n-2]?
        k+=1
    
    autovalores = np.diag(A)

    return autovalores, V

#função que constroi matriz "densa" a partir de matriz diagonal para depois sus autovalores serem calculados
def constroi_matriz(n): 
    autovalores = np.arange(1,n+1)
    D = np.diag(autovalores) #matriz diagonal com os autovalores
    B = np.random.rand(n,n) #matriz que será fatorada em Q e R
    Q, _ = scipy.linalg.qr(B) #Q é ortogonal
    A = Q@D@(Q.T) #matriz final que é cheia

    return A

n = 5
max_iter = 1000
tol = 1e-6

# calculando autovalores e autovetores pelo método de francis

autovalores_francis, autovetores_francis = algoritmo_francis(n, max_iter, tol)
print("Autovalores calculados pelo método de Francis: ", autovalores_francis)
print("Autovetores calculados pelo método de Francis: ", autovetores_francis)

# calculando autovalores e autovetores pelo método scipy.linalg.eigh
A = constroi_matriz(n)
autovalores_scipy, autovetores_scipy = scipy.linalg.eigh(A)
print("Autovalores calculados com scipy: ", autovalores_scipy)
print("Autovetores calculados com scipy: ", autovetores_scipy)

