# Roel Adrián De la Rosa Castillo
# A01197595
# Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución. - Entrega intermedia

# Importamos las librerías necesarias
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargamos nuestro set de datos
data = pd.read_csv('Iris.csv')

# Transformamos los datos para que puedan ser procesados
data = data.drop(['Id'], axis = 1)
X = data.drop(['Species'], axis = 1).to_numpy()
Y = data['Species'].to_numpy()

# Separamos entre set de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=26)

# Hacemos el modelo
model = tree.DecisionTreeClassifier()

# Se entrena el modelo con nuestros datos de entrenamiento
model.fit(X_train, Y_train)

# Se grafica como es que el modelo toma sus decisiones
tree.plot_tree(model)
plt.show()

# Se predice a partir del set de prueba
y_pred = model.predict(X_test)

# Se evalua el modelo y se gráfica una matriz de confusión de los datos que han sido predecidos por el modelo
print("La precision de nuestro modelo es: ", accuracy_score(Y_test, y_pred))
sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True)
plt.title('Las categorias son: Iris-setosa, Iris-versicolor, Iris-virginica')
plt.show()
