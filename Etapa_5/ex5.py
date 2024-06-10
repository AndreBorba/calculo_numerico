import numpy as np
import matplotlib.pyplot as plt

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
    soma = 0 # a soma começa assim pois f(a) e f(b) são somados apenas uma vez
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


def f(x):
    return 4/(1+x**2)

intervalo_teste = np.arange(2,600, 2) # array com diferentes valores a serem usados como subintervalos entre os limites de integração
# listas para armazenar os erros
erros_ptMedio = []
erros_trapezio = []
erros_simpson = []

for i in intervalo_teste:
    integral_ptMedio = ponto_medio(f, 0, 1, i)
    integral_trapezio = trapezio(f, 0, 1, i)
    integral_simpson = simpson(f, 0, 1, i)

    # erros de cálculo referentes a cada método
    erro1 = abs(np.pi - integral_ptMedio)
    erro2 = abs(np.pi - integral_trapezio)
    erro3 = abs(np.pi - integral_simpson)

    # adicionando erros às listas de erros
    erros_ptMedio.append(erro1)
    erros_trapezio.append(erro2)
    erros_simpson.append(erro3)

# plotando o gráfico de erro em função do número de subintervalos
plt.loglog(intervalo_teste, erros_ptMedio, label = 'Método do ponto médio')
plt.loglog(intervalo_teste, erros_trapezio, label = 'Método do trapézio')
plt.loglog(intervalo_teste, erros_simpson, label = 'Método de Simpson')
plt.title("Erro versus número de subintervalos")
plt.xlabel("Subintervalos")
plt.ylabel("Erro")
plt.grid()
plt.legend()
plt.show()


