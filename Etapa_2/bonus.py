import numpy as np
import matplotlib.pyplot as plt

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
    CH_formula1 = 2.3 + 0.1 * (x - 1)**2
    CV_formula1 = 1.8 + 0.2 * (x - 1)**2

    CH_formula2 = 2.3 + 10 * np.exp(-(x - 5)**2)
    CV_formula2 = 1.8 + 10 * np.exp(-(x - 5)**2)

    # calculando a potencia para as duas fórmulas:
   
    m, n = 9, 8
    nv, nc, conec, C, _ = GeraRede(m, n, CH_formula1, CV_formula1)
    pressure_formula1 = SolveNetwork(conec, C, n*m - 1, 0, 3)
    K_formula1 = escreve_mtx_diagonal(C)
    D_formula1 = escreve_mtx_D(C, conec)
    #transpostas
    transposta_pressure_formula1 = pressure_formula1.transpose()
    transposta_D_formula1 = D_formula1.transpose()
    potencia_formula1 = transposta_pressure_formula1 @ (transposta_D_formula1 @ K_formula1 @ D_formula1) @ pressure_formula1

    nv, nc, conec, C, _ = GeraRede(m, n, CH_formula2, CV_formula2)
    pressure_formula2 = SolveNetwork(conec, C, n*m - 1, 0, 3)
    K_formula2 = escreve_mtx_diagonal(C)
    D_formula2 = escreve_mtx_D(C, conec)
    transposta_pressure_formula2 = pressure_formula2.transpose()
    transposta_D_formula2 = D_formula2.transpose()
    potencia_formula2 = transposta_pressure_formula2 @ (transposta_D_formula2 @ K_formula2 @ D_formula2) @ pressure_formula2

    return potencia_formula1, potencia_formula2


# proximas linhas para plotar a potência para diferentes valores de x
x_values = np.linspace(1, 20, 100)
valores_potencia_formula1 = []
valores_potencia_formula2 = []

for x in x_values:
    potencia1, potencia2 = calcula_potencia(x)
    valores_potencia_formula1.append(potencia1)
    valores_potencia_formula2.append(potencia2)
    

plt.plot(x_values, valores_potencia_formula1, label="Fórmula 1")
plt.plot(x_values, valores_potencia_formula2, label="Fórmula 2")
plt.axhline(y=6, color='r', linestyle='-.', label="Potência = 6")
plt.xlabel("Valor de x")
plt.ylabel("Potência dissipada")
plt.title("Potência dissipada em função de x")
plt.legend()
plt.grid()
plt.show()
