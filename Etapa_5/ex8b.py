import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x*np.exp(-x)*np.cos(2*x)

def derivada_adiantada(f, x, h):
    return (f(x+h) - f(x))/h

def derivada_atrasada(f, x, h):
    return (f(x) - f(x-h))/h

def derivada_centrada(f, x, h):
    return (f(x + h) - f(x - h))/(2*h)

valores_h = [0.2, 0.1, 0.05, 0.025]
intervalo = np.linspace(0, np.pi, 500)

resultados_plot = []

for h in valores_h:
    resultados_adiantada = []
    resultados_atrasada = []
    resultados_centrada = []
    for x in intervalo:
        derivada1 = derivada_adiantada(f,x,h)
        derivada2 = derivada_atrasada(f,x,h)
        derivada3 = derivada_centrada(f,x,h)

        resultados_adiantada.append(derivada1)
        resultados_atrasada.append(derivada2)
        resultados_centrada.append(derivada3)

    resultados_plot.append(resultados_adiantada)
    resultados_plot.append(resultados_atrasada)
    resultados_plot.append(resultados_centrada)

plt.figure(figsize=(10,15))

plt.subplot(4,1,1)
plt.plot(intervalo, resultados_plot[0], label='Derivada Adiantada')
plt.plot(intervalo, resultados_plot[1], label='Derivada Atrasada')
plt.plot(intervalo, resultados_plot[2], label='Derivada Centrada')
plt.xlabel('x')
plt.ylabel('Derivada de f')
plt.grid()
plt.legend()
plt.title('Aproximação da derivada com h = 0.2')

plt.subplot(4,1,2)
plt.plot(intervalo, resultados_plot[3], label='Derivada Adiantada')
plt.plot(intervalo, resultados_plot[4], label='Derivada Atrasada')
plt.plot(intervalo, resultados_plot[5], label='Derivada Centrada')
plt.xlabel('x')
plt.ylabel('Derivada de f')
plt.grid()
plt.legend()
plt.title('Aproximação da derivada com h = 0.1')

plt.subplot(4,1,3)
plt.plot(intervalo, resultados_plot[6], label='Derivada Adiantada')
plt.plot(intervalo, resultados_plot[7], label='Derivada Atrasada')
plt.plot(intervalo, resultados_plot[8], label='Derivada Centrada')
plt.xlabel('x')
plt.ylabel('Derivada de f')
plt.grid()
plt.legend()
plt.title('Aproximação da derivada com h = 0.05')

plt.subplot(4,1,4)
plt.plot(intervalo, resultados_plot[9], label='Derivada Adiantada')
plt.plot(intervalo, resultados_plot[10], label='Derivada Atrasada')
plt.plot(intervalo, resultados_plot[11], label='Derivada Centrada')
plt.xlabel('x')
plt.ylabel('Derivada de f')
plt.grid()
plt.legend()
plt.title('Aproximação da derivada com h = 0.025')

# essas linhas são para ajustar espaçamento entre os gráficos
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.5)

plt.show()