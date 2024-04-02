import numpy as np
import matplotlib.pyplot as plt
import time

def vetor_escalar(n):
    a = np.random.rand(n)   # gera vetores aleatórios a e b de dimensão n
    b = np.random.rand(n)

    alpha = np.random.rand() # Gera escalares aleatórios alpha e beta
    beta = np.random.rand()
   
    c = alpha * a + beta * b
    return c

def potencia_matriz(n, m): #funçao que calcula a m-ésima potencia de uma matriz 
    A = np.random.rand(n, n) # gera uma matriz aleatória A de dimensão n x n
    P = A
    for _ in range(m):  # elevando a matriz à potencia m
        P = P @ A
    return P

if __name__ == "__main__":
    # listas para armazenar os tempos demorados para os cálculos
    times_vetor = []
    times_matriz = [[] for _ in range(3)]  # 3 valores de m: 2, 3, 4
    # valores de n para o vetor e para a matriz
    valores_n_vetor = [10**i for i in range(4, 9)]
    valores_n_matriz = [1000, 2000, 3000]

    # calcula o tempo para o cálculo do vetor para diferentes valores de n
    for n in valores_n_vetor:
        ti = time.time()
        vetor_escalar(n)
        tempo = time.time() - ti
        times_vetor.append(tempo)
        print(f"Tempo para vetor com n = {n}: {tempo:.6f} s")

    # calcula o tempo para a matriz para diferentes valores de n e m
    for n in valores_n_matriz:
        for j, m in enumerate([2, 3, 4]):
            ti = time.time()
            potencia_matriz(n, m)
            tempo = time.time() - ti
            times_matriz[j].append(tempo)
            print(f"Tempo para matriz com n = {n} e m = {m}: {tempo:.6f} s")

    # plotagem: tamanho da imagem
    plt.figure(figsize=(15, 6))

    # plotando tempo para vetor em escala linear
    plt.subplot(1, 2, 1)
    plt.plot(valores_n_vetor, times_vetor, marker='o')
    plt.xlabel('Dimensão (n)')
    plt.ylabel('Tempo de Cálculo (s)')
    plt.title('Escala Linear')
    plt.grid(True)
    plt.xticks(fontsize=8)  # aqui define o tamanho da fonte dos números no eixo x

    # plotando o tempo para vetor em escala logarítmica
    plt.subplot(1, 2, 2)
    plt.plot(valores_n_vetor, times_vetor, marker='o')
    plt.xlabel('Dimensão (n)')
    plt.ylabel('Tempo de Cálculo (s)')
    plt.title('Escala Logarítmica')
    plt.grid(True)
    plt.xticks(fontsize=8)  
    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()  # ajusta automaticamente os espaços entre os subplots, não deixa as imagens se sopreborem
    plt.show()

    # aqui vem o plot do grfico do calculo da matriz em escala linear
    for j, m in enumerate([2, 3, 4]):
        plt.plot(valores_n_matriz, times_matriz[j], marker='o', label=f'm={m}')

    plt.xlabel('Dimensão (n)')
    plt.ylabel('Tempo de Cálculo (s)')
    plt.title('Tempo de Cálculo da Matriz em Função da Dimensão n (Escala Linear)')
    plt.grid(True)
    plt.legend()
    plt.xticks(fontsize=8) 
    plt.tight_layout()  
    plt.show()

    # plot do tempo para matriz em escala logarítmica
    plt.figure(figsize=(10, 5))
    for j, m in enumerate([2, 3, 4]):
        plt.plot(valores_n_matriz, times_matriz[j], marker='o', label=f'm={m}')

    plt.xlabel('Dimensão (n)')
    plt.ylabel('Tempo de Cálculo (s)')
    plt.title('Tempo de Cálculo da Matriz em Função da Dimensão (Escala Logarítmica)')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=8)  
    plt.tight_layout()  
    plt.show()



# def vetor_escalar(n):
#     a = np.random.rand(n)
#     b = np.random.rand(n)

#     alpha = np.random.uniform(10.0, 20.0)
#     beta = np.random.uniform(10.0, 20.0)

#     c = alpha*a + beta*b

#     return c

# def potencia_matriz(n, m):
#     A = np.random.rand(n,n)
#     P = A
#     for _ in range(m):
#         P = P@A
#     return P

# if __name__ == "__main__":

#     times = []

#     valores_n = []

#     for i in range(4):
#         n = 10**(i+5)
#         valores_n.append(n)
#         ti = time.time()
#         vetor_escalar(n)
#         tempo = time.time() - ti
#         times.append(tempo)
#         print("t(",i,"): ", tempo )

    
#     plt.plot(valores_n, times, marker='o')
#     plt.xlabel('Dimensão (n)')
#     plt.ylabel('Tempo de Cálculo (s)')
#     plt.title('Tempo de Cálculo em Função da Dimensão')
#     plt.grid(True)
#     plt.show()

    # eixo_x = np.array[]

    # xpoints = np.array(times)
    # ypoints = np.array(ps)

    # plt.plot(xpoints, ypoints)
    # plt.show() 


    # ti = time.time()
    # c = vetor_escalar(100000)
    # t1 = time.time() - ti

    # ti = time.time()
    # c = vetor_escalar(1000000)
    # t2 = time.time() - ti

    # ti = time.time()
    # c = vetor_escalar(10000000)
    # t3 = time.time() - ti

    # ti = time.time()
    # c = vetor_escalar(10000000)
    # t4 = time.time() - ti

    # ti = time.time()
    # c = vetor_escalar(10000000)
    # t5 = time.time() - ti

    # ti = time.time()
    # c = vetor_escalar(100)
    # ta = time.time() - ti

    # ti = time.time()
    # c = vetor_escalar(1000)
    # tb = time.time() - ti

    # print('Tempo 10^5: ', t1)
    # print('Tempo 10^6: ', t2)
    # print('Tempo 10^7: ', t3)
    # print('Tempo 10^8: ', t4)
    # print('Tempo 10^9: ', t5)
    # print('Tempo 10^2: ', ta)
    # print('Tempo 10^3: ', tb)

    # xi = 10**5
    # xf = 10**9
    # N = 4

    # x = ps
    # y = times

    # plt .plot(x, y)
    # plt.show()
    


