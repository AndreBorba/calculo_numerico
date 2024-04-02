import matplotlib.pyplot as plt
import networkx as nx   # biblioteca para grafos
import numpy as np

class Grafo:

    def __init__(self):
        self.nos = {}          # criando um dicionario que vai armazenar os nós e as coordenadas associadas a cada nó
        self.matriz_adj = [[]]   # criando a matriz adjacencia, que é a matriz que contém as arestas

    def add_no(self, no, x, y):
        self.nos[no] = (x,y)   # adicionando a chave 'no' e seu valor (x,y) ao dicionario dos nos
        self.atualiza_matriz() # atualizando a dimensao da matriz das arestas
    
    def add_aresta(self, no1, no2):   # insere uma aresta, assumindo que no1 e no2 já existem
        if no1 in self.nos and no2 in self.nos:
            pos1 = list(self.nos.keys()).index(no1)  # nesta linha e na abaixo desta estou pegando o índice
            pos2 = list(self.nos.keys()).index(no2)  # de cada nó. criei uma lista com os nós e peguei o indice do nó dentro da lista

            self.matriz_adj[pos1][pos2] = 1 # definindo que existe conexão entre no1 e no2
            self.matriz_adj[pos2][pos1] = 1 # essa linha indica que o grafo é não direcionado

    def del_no(self, no):  
        if no in self.nos:
            pos = list(self.nos.keys()).index(no)  
            del self.nos[no]  
            self.matriz_adj = np.delete(self.matriz_adj, pos, axis=0)  # remove a linha correspondente à posição pos
            self.matriz_adj = np.delete(self.matriz_adj, pos, axis=1)  # remove a coluna correspondente à posição pos

    def del_aresta(self, no1, no2):
        if no1 in self.nos and no2 in self.nos:
            pos1 = list(self.nos.keys()).index(no1)  # nesta linha e na abaixo desta estou pegando o índice
            pos2 = list(self.nos.keys()).index(no2)  # de cada no dentro do dicionario

            self.matriz_adj[pos1][pos2] = 0  # retirando as arestas
            self.matriz_adj[pos2][pos1] = 0  
            

            # da linha 42 à linha 58, está a lógica para apagar os nós caso não estejam associados
            # a nenhuma outra aresta

            flag = False
            for coluna in self.matriz_adj[pos1]:
                if coluna == 1:
                    flag = True
                    break  # se encontrou uma aresta, não precisa continuar procurando
            
            if not flag:
                self.del_no(no1)

            flag = False
            for coluna in self.matriz_adj[pos2]:
                if coluna == 1:
                    flag = True
                    break
            
            if not flag:
                self.del_no(no2)

    def atualiza_matriz(self):
        n = len(self.nos)
        matriz = self.matriz_adj

        if n == len(matriz):  # a matriz já está do tamanho certo, não precisa ser atualizada
            return

        # aqui a matriz será redimensionada mas serão mantiads as conexões existentes
        nova_matriz_adj = np.zeros(shape=(n,n))
        for i in range(min(n, len(matriz))):
            for j in range(min(n, len(matriz[i]))):
                nova_matriz_adj[i][j] = matriz[i][j]   # copiando as entradas da matriz original para a nova matriz
        self.matriz_adj = nova_matriz_adj

    def plota_grafo(self):
        grafo = nx.Graph()
        for no, coordenadas in self.nos.items():
            grafo.add_node(no, pos=coordenadas)  # o argumento 'pos' é responsável por plotar o nó em um plano cartesiano
        
        arestas = self.cria_lista_arestas()   # lista das arestas
        grafo.add_edges_from(arestas)  # adiciona varias arestas de uma só vez, recebe uma lista de tuplas (no1, no2) como argumento
        pos = nx.get_node_attributes(grafo, 'pos')
        nx.draw(grafo, pos, with_labels=True, node_color='yellow', node_size=500)  # desenha o grafo
        plt.show()

    def cria_lista_arestas(self):   # função para gerar uma lista contendo as arestas e serve depois para plotar o grafo
        arestas = []
        n = len(self.nos)
        lista_nos = list(self.nos.keys())  # criando uma lista com todos os nós
        for i in range(n):
            for j in range(i + 1, n):
                if self.matriz_adj[i][j] == 1:
                    arestas.append((lista_nos[i], lista_nos[j]))  # adicionando tuplas à lista arestas, essas tuplas dizem quais nós possuem ligação entre si
        
        return arestas
    

meu_grafo = Grafo()

meu_grafo.add_no(0, 1, 1)
meu_grafo.add_no(1, 2, 2)
meu_grafo.add_no(2, 3, 2.1)
meu_grafo.add_no(3, 2, 0.8)
meu_grafo.add_no(4, 3, 0.7)
meu_grafo.add_no(5, 4, 0.5)
meu_grafo.add_no(6, 4.2, 2.3)
meu_grafo.add_no(7, 5, 1.1)
meu_grafo.add_no(8, 6, 2.3)
meu_grafo.add_no(9, 6.1, 0.6)

meu_grafo.add_aresta(0,1)
meu_grafo.add_aresta(1,3)
meu_grafo.add_aresta(1,2)
meu_grafo.add_aresta(3,4)
meu_grafo.add_aresta(4,2)
meu_grafo.add_aresta(2,6)
meu_grafo.add_aresta(6,5)
meu_grafo.add_aresta(6,7)
meu_grafo.add_aresta(7,8)
meu_grafo.add_aresta(7,5)
meu_grafo.add_aresta(8,9)

meu_grafo.plota_grafo()



