import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#criando o dataset
dados = pd.read_csv('https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv')
#convertendo de pandas pra um array numpy
dados = dados.to_numpy()

#função que separa os dados em 2 arrays, um para trabalhos finalizados e outro para não finalizados e em seguida os plota em cores diferentes
def plot_data(dados_x, dados_y):
    arr_filter = []
    for elem in dados_y:
        if elem == 0:
            arr_filter.append(True)
        else:
            arr_filter.append(False)
    dados_0 = dados_x[arr_filter]
    dados_1 = dados_x[~np.array(arr_filter)]

    plt.scatter(dados_0[0:,0], dados_0[0:, 1], label='Não concluídos')
    plt.scatter(dados_1[0:, 0], dados_1[0:, 1], label='Concluídos')
    plt.title('Relação Dados de entrada')
    plt.xlabel('Horas esperadas')
    plt.ylabel('Preço')
    plt.legend()
    plt.show()

x = dados[0:, 1:]
y = dados[0:, 0]

#definindo tamanho do conjunto de teste
test_size = 0.2
#separando os conjuntos de treino e de teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

plot_data(x_test, y_test)

#Criando o classificador
clf = DecisionTreeClassifier()
#Treinando o classificador com o conjunto de treino
clf.fit(x_train, y_train)
#Testando e medindo acurácia com o conjunto de teste
print(f'Acurácia: {clf.score(x_test, y_test)}')