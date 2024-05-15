import numpy as np 
import pandas as pd

# Método de Jacobi
N = [11, 21, 41]  # Número de pontos em cada direção
erros = [1.0e-5, 1.0e-8]
dados_tabela = []

for n in N:
    for erro in erros:
        iteracoes = []
        for _ in range(10):  # Rodar 10 vezes para calcular uma média
            Told = np.zeros(shape=(n, n))
            Tnew = np.zeros(shape=(n, n))

            # Temperaturas nas bordas
            Told[0, :] = 0.0  # TL
            Told[n-1, :] = 0.0  # TR
            Told[:, 0] = 0.0  # TB
            Told[:, n-1] = 20.0  # TT

            # Loop de iterações
            Nmax = 10000
            Tnew = Told.copy()
            for iter in range(Nmax):
                Tnew[1:n-1, 1:n-1] = 0.25 * (Told[2:n, 1:n-1] + Told[0:n-2, 1:n-1] +
                                              Told[1:n-1, 2:n] + Told[1:n-1, 0:n-2])
                error = np.linalg.norm(Tnew - Told)

                if error < erro:
                    iteracoes.append(iter)
                    break
                Told = Tnew.copy()

        media_iteracoes = np.mean(iteracoes)
        dados_tabela.append([n, erro, media_iteracoes])

# Criando DataFrame para a tabela
df = pd.DataFrame(dados_tabela, columns=['N', 'Tolerância', 'Iterações Médias'])

print(df)
