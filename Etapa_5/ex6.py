import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def Assembly(conec, C):
    lista_mtx = []
    for i in conec:
        for j in i:
            lista_mtx.append(j)

    nv = max(lista_mtx) + 1
    nc = len(conec)
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
    nc = len(C)
    K = np.zeros(shape=(nc, nc))
    for i in range(nc):
        K[i,i] = C[i]
    return K

def escreve_mtx_D(C, conec):
    nv = np.max(conec) + 1
    nc = len(C)
    D = np.zeros((nc, nv))

    for k in range(nc):
        D[k, conec[k, 0]] = 1
        D[k, conec[k, 1]] = -1

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
    
    for j in range(ny):
        for i in range(nx-1):
            k = j*(nx-1) + i
            conec[k,0] = j*nx + i
            conec[k,1] = j*nx + i+1
            C[k] = CH

    for i in range(nx):
        for j in range(ny-1):
            k = (nx-1)*ny + j*nx + i
            conec[k,0] = i + j*nx
            conec[k,1] = i + (j+1)*nx
            C[k] = CV

    return nv, nc, conec, C, coord

def calcula_potencia(x):
    # condutividades de acordo com as fórmulas
    CH = 2.3 + 10 * np.exp(-(x - 5)**2)
    CV = 1.8 + 10 * np.exp(-(x - 5)**2)

    # calculando a potencia para as duas fórmulas:
    m, n = 9, 8
    nv, nc, conec, C, _ = GeraRede(m, n, CH, CV)
    pressure = SolveNetwork(conec, C, n*m - 1, 0, 3)
    K = escreve_mtx_diagonal(C)
    D = escreve_mtx_D(C, conec)
    transposta_pressure = pressure.transpose()
    transposta_D = D.transpose()
    potencia = transposta_pressure @ (transposta_D @ K @ D) @ pressure

    return potencia



######## AGORA VEM A PARTE DAS INTEGRAIS ##########

'''
É bom perceber que, nesse exemplo, nós temos a função exata 
que queremos avaliar (potência consumida pela bomba). TEmos a função exata
pois ela depende de x, visto que as condutâncias vertical e horizontal dependem
explicitamente de x. Caso tivessemos apenas dados aleatórios (advindos de algum
experimento, por exemplo), seria necessário realizar uma interpolação para
definir uma função de x hipotética e, assim, poder realizar os cálculos de 
integral.
'''

# Regra do ponto médio
def ponto_medio(f, a, b, n):
    h = (b-a)/n # tamanho de cada subintervalo, geometricamente é um lado do retângulo
    soma = 0
    for i in range(n):
        xi = a + h*i + h/2  # ponto médio de cada subintervalo analisado nas iterações
        soma += f(xi)
    
    integral = h*soma

    return integral

# Regra do trapézio
def trapezio(f, a, b, n):
    h = (b-a)/n # tamanho de cada subintervalo, geometricamente é a altura do trapézio
    soma = 0 
    for i in range(n):
        x1 = a + h*i
        x2 = x1 + h
        soma += f(x1) + f(x2)
    
    integral = h/2 * soma

    return integral

# Regra de Simpson
def simpson(f, a, b, n):
    h = (b-a)/n  # tamanho de cada subintervalo
    soma = 0
    for i in range(1, n + 1):  # ajustando o loop para começar de 1 até n
        x1 = a + h * i  # x(k)
        x2 = a + h * (i - 1)  # x(k-1)
        x_medio = (x1 + x2) / 2  # ponto médio do subintervalo
        soma += f(x2) + 4 * f(x_medio) + f(x1)
    integral = (h / 6) * soma

    return integral

intervalos_teste = [2,4,6,8,10]

# lista para armazenar os resultados
resultados = []

for i in intervalos_teste:
    integral_ptMedio = ponto_medio(calcula_potencia, 1, 10, i)
    integral_trapezio = trapezio(calcula_potencia, 1, 10, i)
    integral_simpson = simpson(calcula_potencia, 1, 10, i)

    resultados.append([i, integral_ptMedio, integral_trapezio, integral_simpson])

# criando a tabela para exibir os resultados
tabela = pd.DataFrame(resultados, columns=["Número de subintervalos", "Integral ponto médio", "Integral trapézio", "Integral Simpson"])
pd.set_option('display.max_columns', None) # ajuste necessário para exibir todas as colunas
print(tabela)


