import numpy as np 
import scipy 

def gera_simetrica(n):
    A = np.random.rand(n,n)
    A = A + A.T
    return A

def algoritmo_francis(n, max_iter, tol):
    k = 0 # representa as iterações
    A = gera_simetrica(n) #matriz que conterá autovalores em sua diagonal principal
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

dimensoes = [10, 50, 100, 200]
max_iter = 1000
tol = 1e-6

for n in dimensoes:
    autovalores, autovetores = algoritmo_francis(n, max_iter, tol)
    print("Autovalores:", autovalores)
    print("Autovetores:", autovetores)