import numpy as np
import time

a = np.random.uniform(0.0, 4.0)  # valor aleatorio de a entre 0 e 4
x0 = 0.1 
sequencia = np.zeros(5001)  # "alocando" espaço para sequencia
sequencia[0] = x0

for i in range(1, 5001):
    sequencia[i] = a*sequencia[i-1]*(1-sequencia[i-1]) # calculo dos valores da sequencia 

# agora vem o cálculo da media sem usar numpy

t0 = time.time()
soma = 0
for i in range(5001):
    soma += sequencia[i]

media = soma/5001
tf = time.time()
print("Tempo demorado para media calculada: ", tf - t0)

# media = sum(sequencia)/len(sequencia)

# agora vem o cálculo da variancia sem usar o numpy

desvio = 0
t0 = time.time()
for i in range(5001):
    desvio += (sequencia[i] - media) ** 2

variancia = desvio/5000  # variancia = desvio/(N-1)
tf = time.time()
print("Tempo demorado para variancia calculada: ", tf - t0)
print("Media calculada: ", media)
print("Variancia calculada: ", variancia)

# agora vem os calculos usando as funções do numpy

t0 = time.time()
media1 = np.mean(sequencia)
tf = time.time()
print("Tempo demorado para media numpy: ", tf - t0)

t0 = time.time()
variancia1 = np.var(sequencia, ddof=1)
tf = time.time()
print("Tempo demorado para variancia numpy: ", tf - t0)

print("Média numpy: ",media1)
print("Variancia numpy: ",variancia1)