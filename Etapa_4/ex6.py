import numpy as np
import scipy.sparse

def ij2n(i, j, N1):
    #converte as coordenadas (i, j) da grade para o índice do vetor correspondente
    return i + j * N1

    #define a função de densidade rho(x, y)
def rho_funcao(x, y):
    return 1 + 0.75 * np.cos(4 * np.pi * x) * np.cos(2 * np.pi * y)

def BuildMatrizesEigen(N1, N2, sigma, e, delta):
    nunk = N1 * N2  

    # Stiffness matrix K: Build it as a sparse matrix 
    d1 = 4.0*np.ones(nunk)
    d2 = -np.ones(nunk-1)
    d3 = -np.ones(nunk-N1)
    K = (sigma/delta**2)*scipy.sparse.diags([d3, d2, d1, d2, d3], [-N1, -1, 0, 1, N1], format='csr')

    # Force the eigenvalues associated to boundary points 
    # to be a big number as compared to fundamental modes
    big_number = 10000
    Iden = big_number*scipy.sparse.identity(nunk, format='csr')

    # Lados verticais
    for k in range(0,N2):
        Ic = ij2n(0,k,N1) # Left
        K[Ic,:], K[:,Ic] = Iden[Ic,:], Iden[:,Ic]

        Ic = ij2n(N1-1,k,N1) # Right
        K[Ic,:], K[:,Ic] = Iden[Ic,:], Iden[:,Ic]
        
    # Lados horizontais
    for k in range(0,N1):
        Ic = ij2n(k,0,N1) # Bottom
        K[Ic,:], K[:,Ic] = Iden[Ic,:], Iden[:,Ic]

        Ic = ij2n(k,N2-1,N1) # Top
        K[Ic,:], K[:,Ic] = Iden[Ic,:], Iden[:,Ic]
                                                 
                                                

    # criação de uma matriz de zeros para a matriz de massa M
    M = scipy.sparse.lil_matrix((nunk, nunk))


    # calculando aa matriz de massa M com base na densidade variável
    for i in range(N1):
        for j in range(N2):
            x = i * delta  # coordenada x do ponto (i, j) na grade, smp positivo
            y = j * delta  # coordenada y do ponto (i, j) na grade, smp positivo
            rho_val = rho_funcao(x, y)  # densidade no ponto (x, y)
            idx = ij2n(i, j, N1)  
            M[idx, idx] = rho_val * e   # ajusta a entrada correspondente na matriz de massa, só mexe na diagonal principal

    return K, M

# Parâmetros
N1 = 41
N2 = 21
L1 = 1.0
L2 = 0.5
# Distancia entre pontos: Cuidado que L1/(N1-1) seja igual a L2/(N2-1)
delta = L1 / (N1-1) 

# Tensão da membrana
sigma = 1.0
rho = 1
e = 1

# Aplicar restrições nas bordas
K, M = BuildMatrizesEigen(N1, N2, sigma, e, delta)
# Exemplo de uso das matrizes K e M
print("Matriz de rigidez K:")
print(K)
print("Matriz de massa M:")
print(M)
