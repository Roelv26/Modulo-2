import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Iris.csv')

data = data.drop(['Id'], axis = 1)
X = data.drop(['Species'], axis = 1).to_numpy()
Y = data['Species'].to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=26)

model = tree.DecisionTreeClassifier()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print("La precision de nuestro modelo es: ", accuracy_score(Y_test, y_pred))
sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True)
plt.title('Las categorias son: Iris-setosa, Iris-versicolor, Iris-virginica')
plt.show()