import numpy as np
import scipy

# A = QD(Q^-1), A é simétrica, Q é ortogonal e D é diagonal

#função que constroi matriz "densa" a partir de matriz diagonal para depois sus autovalores serem calculados
def constroi_matriz(n): 
    autovalores = np.arange(1,n+1)
    D = np.diag(autovalores) #matriz diagonal com os autovalores
    B = np.random.rand(n,n) #matriz que será fatorada em Q e R
    Q, _ = scipy.linalg.qr(B) #Q é ortogonal
    A = Q@D@(Q.T) #matriz final que é cheia

    return A

n = 5
A = constroi_matriz(n)
print(A)

autovalores_calculados, autovetores_calculados = scipy.linalg.eigh(A)
print("Autovalores calculados com scipy: ", autovalores_calculados)
print("Autovetores calculados com scipy: ", autovetores_calculados)
