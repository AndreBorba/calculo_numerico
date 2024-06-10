import numpy as np
import matplotlib.pyplot as plt

# Definindo a função f
def f(x):
    return x * np.exp(-x) * np.cos(2 * x)

# Derivada analítica de f
def derivada_f(x):
    return ((np.exp(-x) - x * np.exp(-x)) * np.cos(2 * x) - 2 * x * np.exp(-x) * np.sin(2 * x))

# Valores de delta (h)
valores_k = np.arange(0, 11)
valores_h = [0.25 / (10 ** k) for k in valores_k]

# Derivada adiantada
def derivada_adiantada(f, x, h):
    return (f(x + h) - f(x)) / h

# Derivada centrada
def derivada_centrada(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# Ponto onde calculamos a derivada
x_bar = np.pi / 2
derivada_analitica = derivada_f(x_bar)

# Listas para armazenar os erros
erros_adiantada = []
erros_centrada = []

for h in valores_h:
    derivada_a = derivada_adiantada(f, x_bar, h)
    derivada_c = derivada_centrada(f, x_bar, h)
    
    erro_adiantada = abs(derivada_analitica - derivada_a)
    erro_centrada = abs(derivada_analitica - derivada_c)
    
    erros_adiantada.append(erro_adiantada)
    erros_centrada.append(erro_centrada)

# Plotando os resultados em escala logarítmica
plt.figure(figsize=(10, 6))
plt.loglog(valores_h, erros_adiantada, label='Erro - Derivada Adiantada', marker='o')
plt.loglog(valores_h, erros_centrada, label='Erro - Derivada Centrada', marker='x')
plt.xlabel('h (Delta)')
plt.ylabel('Erro')
plt.title('Erro das Fórmulas de Diferenciação Numérica em x = π/2')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()


'''
Percebe-se que para valores muito pequenos de h, o erro aumenta.
'''