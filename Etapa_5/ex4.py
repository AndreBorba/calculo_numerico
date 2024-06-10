import numpy as np
import matplotlib.pyplot as plt

# Carregar os dados do arquivo
data = np.loadtxt('/home/andre/Documentos/semestre3/calculo_numerico/Etapa_5/data.txt')

# Função para aplicar "equações normais"
def equacao_normal(X, y):
    linhas_X = X.shape[0]
    coluna_1 = np.ones((linhas_X, 1)) # criando matriz coluna composta por "1" para depois juntar com a matriz X e formar a matriz A
    A = np.c_[coluna_1, X]
    constantes = np.linalg.inv(A.T @ A) @ (A.T @ y) # da equação normal que está no material de aula: multiplicar PELA ESQUERDA a inversa de (A.T @ A) em ambos os lados. Assim aparece a matriz de constantes
    return constantes

# Função para prever os valores usando os coeficientes obtidos. 
# EXISTE UMA FUNÇÃO PRONTA PARA FAZER ISSO, DA BIBLIOTECA sklearn.linear_model, ela é usada em machine learning
def predict(X, constantes):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b @ constantes

# Caso 1: c(3) = k1 + k2c(1)
#X1 = data[:, 0].reshape(-1, 1) reshape para deixar X1 na forma de matriz coluna (argumento -1 pega o número total de elementos do array e o tranforma no número de linhas da matriz)
X1 = data[:, 0]
y1 = data[:, 2]
constantes_1 = equacao_normal(X1, y1)
k0, k1 = constantes_1
x1_plot = np.linspace(min(X1), max(X1), 100)
y1_plot = np.polyval([k1, k0], x1_plot)

# Plotando os resultados do Caso 1
plt.scatter(X1, y1, color='red', label='Dados originais')
plt.plot(x1_plot, y1_plot, color='green', label='Ajuste Linear')
plt.xlabel('c1 (x)')
plt.ylabel('c3 (y)')
plt.title('Caso 1: c(3) = k1 + k2c1')
plt.legend()
plt.show()

# Caso 2: c(3) = k1 + k2c1 + k3c1^2 
X2 = np.c_[data[:, 0], data[:, 0]**2] # pegando dados da coluna 1 do .txt e fazendo matriz com linhas e colunas do tipo xi xi^2
y2 = data[:, 2] # dados da coluna 3 são o "y" da equação
constantes_2 = equacao_normal(X2, y2)
k0, k1, k2 = constantes_2
x2_plot = np.linspace(min(data[:, 0]), max(data[:, 0]), 100) # valores mínimo e máximo de "x", que são mínimo e máximo da coluna 1 do .txt
y2_plot = np.polyval([k2, k1, k0], x2_plot)

# Plotando os resultados do Caso 2
plt.scatter(data[:, 0], y2, color='red', label='Dados originais')
plt.plot(x2_plot, y2_plot, color='green', label='Ajuste Polinomial (quadrático)')
plt.xlabel('c1 (x)')
plt.ylabel('c3 (y)')
plt.title('Caso 2: c(3) = k1 + k2c(1) + k3c1²')
plt.legend()
plt.show()

# Caso 3: c(4) = k1 + k2c(1) + k3c(2)
# perceber que esse caso é espacial 3d
X3 = np.c_[data[:, 0], data[:, 1]] # primeira coluna de X3 é de varáiveis compostas pela coluna 1 do .txt e segunda coluna de X3 é composta das variáveis da segunda coluna do .txt
y3 = data[:, 3] # "y" da equação são os valores da quarta coluna do .txt. Na verdade esse y pode ser considerado como "z"
constantes_3 = equacao_normal(X3, y3) 
k0, k1, k2 = constantes_3

# Gerando uma malha de pontos para o plano

# meshgrid vai retornar duas matrizes: a primeira replica os valores de x para todas as posições em y. A segunda replica os valores de y para todas as posições em x. Nesta explicação, x e y são os parâmetros da função: np.meshgrid(x,y)
x3_x_plot, x3_y_plot = np.meshgrid(np.linspace(min(data[:, 0]), max(data[:, 0]), 100),
                                   np.linspace(min(data[:, 1]), max(data[:, 1]), 100))
X3_plot = np.c_[x3_x_plot.ravel(), x3_y_plot.ravel()] # ravel "achata" uma matriz, faz virar unidimensional
y3_plot = predict(X3_plot, constantes_3).reshape(x3_x_plot.shape) # precisa ter a mesma dimensão de x3_x_plot e x3_y_plot para usar na função plot_surface
# talvez fosse mais claro deixar a variável acima declarada como z3_plot, uma vez que ela representa os valores do eixo z de um par ordenado (x,y)
# essa função predict está explicada nas notas do tablet

# Plotando os resultados do Caso 3
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], y3, color='red', label='Dados originais')
ax.plot_surface(x3_x_plot, x3_y_plot, y3_plot, color='green', alpha=0.5) #plot_surface parece com plot, só que um par (x,y) agora tem um valor de z associado a eles, que nesse caso são os valores contidos na matriz y3_plot
ax.set_xlabel('c1 (x)')
ax.set_ylabel('c2 (y)')
ax.set_zlabel('c4 (z)')
plt.title('Modelo 3: c(4) = k1 + k2c1 + k3c2')
plt.legend()
plt.show()
