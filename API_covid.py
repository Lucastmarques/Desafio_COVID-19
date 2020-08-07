import pandas as pd 
import numpy as np 
import sys 
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

### Importa base de dados
data = pd.read_csv('Dataset_finalizado.csv')

# Guarda o valor informado no prompt como cidade e data
city_request = sys.argv[2]
date_request = sys.argv[1]

city_request = city_request.upper()

# Verifica se a cidade coincide com outra cidade em nossa base de dados
if(not (pd.isnull(data.loc[data.city == city_request].city).values[0])):
    # Recebe as informações da nossa base de dados e acrescenta em requested_info
    requested_info = data.loc[data.city == city_request]
    requested_info.date = date_request
    requested_info = requested_info.iloc[[0], :]
    
    # Concatena a nossa base de dados com os dados de previsão (requested_info)
    data_concat = pd.concat([requested_info, data], ignore_index=True).dropna()

    # Separa X e Y
    X = data_concat.iloc[:, :].drop(columns=['last_available_confirmed']).values
    Y = data_concat.iloc[1:, 2].values

    # Transforma variáveis categóricas em numéricas
    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:,0])
    X[:, 1] = labelencoder_X.fit_transform(X[:,1])

    # Faz com que cada cidade tenha o mesmo peso adicionando uma coluna por cidade
    onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1])], remainder='passthrough')
    X = onehotencorder.fit_transform(X)

    # Padroniza os valores
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    scaler_Y = StandardScaler()
    Y = scaler_Y.fit_transform(Y.reshape(-1,1))

    # Separa a primeira linha (prever) das demais (previsores para treino)
    X_prev = [X[0, :]]
    X_train = X[1:, :]

    # Inicializa e treina regressor
    regressor = SVR(kernel='rbf')
    regressor.fit(X_train, Y)

    # Prever o número de casos para argv[2] no dia argv[1]
    previsao = regressor.predict(X_test)
    # Coloca o valor previsto na escala normal
    previsao = scaler_Y.inverse_transform(previsao)
    print(previsao[0])

else:
    # Se a cidade não for achada, retorn mensagem de 'erro'
    print('Nome informado não condiz com nenhum outro nome em nossas bases de dados.')