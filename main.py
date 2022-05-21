import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

dados = pd.read_csv('https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv')
dados = dados.to_numpy()

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


x_train, x_test, y_train, y_test = train_test_split(x, y)

plot_data(x_test, y_test)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 2), random_state=1, max_iter=20000)
clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))