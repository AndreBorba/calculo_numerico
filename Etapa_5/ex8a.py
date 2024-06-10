import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x*np.exp(-x)*np.cos(2*x)

# a derivada da função f foi calculada e seu resultado está a seguir:
def derivada_f(x):
    return ((np.exp(-x) - x*np.exp(-x))*np.cos(2*x) - 2*x*np.exp(-x)*np.sin(2*x))

# intervalo de valores para calcular a derivada
intervalo = np.linspace(0, np.pi, 500)

# lista para armazenar resultados
resultados = []

for x in intervalo:
    derivada = derivada_f(x)
    resultados.append(derivada)

plt.plot(intervalo, resultados, label='Derivada da função f')
plt.xlabel('x')
plt.ylabel('Derivada de f')
plt.title('Derivada da função f com relação a "x"')
plt.legend()
plt.grid()
plt.show()