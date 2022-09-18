# Importamos las librerías necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Función que realiza la regresión lineal
def lin_reg(X, Y, learning_rate, iteration):
    m = Y.size
    theta = np.zeros((2,1))
    cost_list = []
    
    for i in range(iteration):
        y_pred = np.dot(X, theta)
        cost = (1/(2*m))*np.sum(np.square(y_pred-Y))
        
        d_theta = (1/m)*np.dot(X.T, y_pred-Y)
        theta = theta - learning_rate*d_theta
        
        cost_list.append(cost)
        
    return theta, cost_list

# leemos y transformamos los datos a un formato que podamos utilizar
df = pd.read_csv('train.csv')
df = df[['age','bmi', 'charges']]
df = df.dropna()
X = (df.drop(['charges'], axis = 1)).to_numpy()
Y = df['charges'].to_numpy()


# Set de entrenamiento y set de prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=420)

# Se entrena la regresión lineal
iteration = 200
learning_rate = 0.0005
theta, cost_list = lin_reg(X_train,y_train, learning_rate=learning_rate, iteration=iteration)

# Evaluamos el modelo
for i in range(X_test.shape[0]):
    print('El modelo predice: ', round(np.dot(X_test[i], theta)[0],2), "| El real es: ",y_test[i], " | La diferencia es de: ", round(abs(round(np.dot(X_test[i], theta)[0],2) - y_test[i])))
