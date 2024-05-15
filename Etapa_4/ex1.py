import numpy as np 
import scipy 
import time
import matplotlib.pyplot as plt

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

#dimensoes = [10, 50, 100, 200]
dimensoes = [i for i in range(10,200,10)]
max_iter = 1000
tol = 1e-6
tempos = []

for n in dimensoes:
    ti = time.time()
    autovalores, autovetores = algoritmo_francis(n, max_iter, tol)
    tf = time.time() - ti
    tempos.append(tf)

print(tempos)

plt.plot(dimensoes, tempos, marker='o')
plt.xlabel('dimensão da matriz (n)')
plt.ylabel('tempo')
plt.grid()
plt.show()