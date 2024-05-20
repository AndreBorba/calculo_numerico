import numpy as np
import scipy.sparse

#converte as coordenadas (i, j) da grade para o índice do vetor correspondente
def ij2n(i, j, N1):
    return i + j * N1

def BuildMatrizesEigenCircular(N1, N2, sigma, rho, e, delta, R):
    nunk = N1 * N2  
    
    #criaando a matriz de rigidez K como uma matriz esparsa
    d1 = 4.0 * np.ones(nunk)  # diagonal principal com valor 4.0
    d2 = -np.ones(nunk - 1)   # diagonais secundárias com valor -1.0
    d3 = -np.ones(nunk - N1)  # diagonais terciárias com valor -1.0
    K = (sigma / delta**2) * scipy.sparse.diags([d3, d2, d1, d2, d3], [-N1, -1, 0, 1, N1],  format='csr')                                               
                                               
    # força os autovalores associados aos pontos de fronteira a serem grandes
    big_number = 10000
    Iden = big_number * scipy.sparse.identity(nunk, format='csr')

    # máscara para identificar os pontos dentro da membrana circular
    mask = np.zeros((N1, N2), dtype=bool)
    for i in range(N1):
        for j in range(N2):
            x = (i - N1 // 2) * delta  # coordenada x do ponto (i, j) na grade
            y = (j - N2 // 2) * delta  # coordenada y do ponto (i, j) na grade
            if np.sqrt(x**2 + y**2) <= R:  # verifica se o ponto está dentro do círculo
                mask[i, j] = True

    # aplicando a máscara na matriz de rigidez K
    # com 2 for ele já passa pela parte vertical e horizontal
    for i in range(N1):
        for j in range(N2):
            if not mask[i, j]:  # se o ponto está fora do círculo
                Ic = ij2n(i, j, N1)
                K[Ic, :] = Iden[Ic, :]  # define as linhas e colunas correspondentes em K
                K[:, Ic] = Iden[:, Ic]

     #Mass matrix: Simple case, multiple of identity
    M = rho * e * scipy.sparse.identity(nunk, format='csr')
    
    return K, M

# Parâmetros
N1 = 100  # Número de pontos horizontais
N2 = 100  # Número de pontos verticais
sigma = 1  # Tensão superficial
rho = 1  # Densidade
e = 1  # Espessura
delta = 0.01  # Espaçamento entre os pontos

K, M = BuildMatrizesEigenCircular(N1, N2, sigma, rho, e, delta, 0.5)

# Análise dos modos de oscilação (cálculo dos autovalores e autovetores)
eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(K, k=20, M=M, which='SM')

# Impressão dos autovalores encontrados
print("Autovalores:", eigenvalues)
print("Autovetores:", eigenvectors)
