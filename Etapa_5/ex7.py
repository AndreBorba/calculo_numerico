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



######## AGORA VEM A PARTE DAS DERIVADAS ##########

'''
É bom perceber que, nesse exemplo, nós temos a função exata 
que queremos avaliar (potência consumida pela bomba). TEmos a função exata
pois ela depende de x, visto que as condutâncias vertical e horizontal dependem
explicitamente de x. Caso tivessemos apenas dados aleatórios (advindos de algum
experimento, por exemplo), seria necessário realizar uma interpolação para
definir uma função de x hipotética e, assim, poder realizar os cálculos de 
derivada.
'''

def derivada_centrada(f, x, h=1e-5):
    return (f(x + h) - f(x - h))/(2*h)

# intervalo de valores de x para calcular a potência e sua derivada
intervalo_x = np.linspace(1, 10, 100)

# calculando a potência e sua derivada para cada valor de x
potencias = np.array([calcula_potencia(x) for x in intervalo_x])
derivadas = np.array([derivada_centrada(calcula_potencia, x) for x in intervalo_x])

# plotando a função da potência e sua derivada
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(intervalo_x, potencias, label='Potência Consumida')
plt.xlabel('x')
plt.ylabel('Potência')
plt.legend()
plt.title('Potência consumida pela bomba em função de "x"')

plt.subplot(2, 1, 2)
plt.plot(intervalo_x, derivadas, label='Derivada da Potência', color='red')
plt.xlabel('x')
plt.ylabel('Derivada da Potência')
plt.legend()
plt.title('Derivada da potência consumida pela bomba com relação a "x"')

plt.tight_layout()
plt.show()

