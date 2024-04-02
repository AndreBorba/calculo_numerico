import numpy as np
import matplotlib.pyplot as plt

x0 = 0.1 # primeiro valor das sequências
valores_a = np.linspace(1, 4, 2500)  # aqui é ajustado o valor de 'a'

X = []    # criando duas listas que vão conter os valores de 'a' (eixo X)
Y = []    # e os elementos das sequências (eixo Y)

for a in valores_a:   # iterando sobre todos os valores de a, definidos na linha 5
  
    s = x0   # sempre inicializando o primeiro elemento de cada sequência como x0

    for i in range(5000):
        s = a * s * (1 - s)  # lei de formação da sequência
        if i >= 4800:    # adicionando os pontos à sequência após ter sido atingida
            X.append(a)  # uma constância / um padrão           
            Y.append(s)

plt.plot(X, Y, ',k', alpha=0.3)

plt.title("Diagrama de bifurcação")
plt.xlabel("Valor de a")
plt.ylabel("Valores da sequência associada ao valor 'a'")
plt.show()

# sn+1​=a⋅sn​⋅(1−sn​)

# Onde:

#     sn​ é a população em uma determinada geração n,
#     a é um parâmetro que representa a taxa de crescimento da população, e
#     sn+1​ é a população na próxima geração.

# A ideia por trás do diagrama de bifurcação é examinar como o comportamento do sistema muda à medida que o parâmetro a é variado.

# A interpretação do diagrama de bifurcação pode ser complexa, mas em geral:

#     Para valores pequenos de a, a população tende a se estabilizar em um valor fixo após algumas gerações.
#     Conforme a aumenta, a população pode oscilar entre dois ou mais valores.
#     Em alguns intervalos específicos de a, podem ocorrer bifurcações, onde o comportamento do sistema muda abruptamente. Isso é representado no diagrama por ramificações ou múltiplos pontos.
#     Conforme a continua a aumentar, a complexidade do comportamento aumenta, com períodos de oscilação, bifurcações e eventual caos.

# Portanto, o diagrama de bifurcação mostra como o comportamento do sistema evolui à medida que o parâmetro aa é variado, oferecendo insights sobre a dinâmica complexa do sistema logístico.