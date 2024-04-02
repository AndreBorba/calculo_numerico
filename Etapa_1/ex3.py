import numpy as np
import matplotlib.pyplot as plt

valores_a = [1, 2, 3.8, 4]
x0 = 0.1
sequencia = np.zeros(5001)
sequencia[0] = x0

# para o gráfico
xpoints = np.arange(0, 5001)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

for i, a in enumerate(valores_a):
    for j in range(1, 5000):
        sequencia[j] = a * sequencia[j - 1] * (1 - sequencia[j - 1])
    
    
    ax = axs[i // 2, i % 2] #aqui definem-se as posições dos gráficos na "matriz imagem" 2x2. // representa o quociente da divisão e % representa o resto da divisão
    ax.grid(True)
    ax.scatter(xpoints, sequencia, 3) #função para plotar apenas os pontos, sem que exista uma linha que os conecte
    ax.set_title('Sequência numérica')
    ax.legend([f'a = {a}'])
    ax.set_xlabel('n')
    ax.set_ylabel('Valores da Sequência')

plt.tight_layout() # função que ajusta os subplots na imagem, não deixa as coisas se sobreporem, deixa tudo bonitinho. É bom usar antes do plt.show() quando tem subplots
plt.show()




