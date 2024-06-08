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
    soma = 0
    for i in range(n):
        x1 = a + h*i
        x2 = x1 + h
        soma += f(x1) + f(x2)
    
    integral = h/2 * soma

    return integral