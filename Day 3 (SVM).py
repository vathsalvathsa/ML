#IMPORTING LIB...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORTING THE DATASET
dataset = pd.read_csv('D:/Position_Salaries.csv')

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

print(X)
print(y)

y = y.reshape(len(y),1)
print(y)

#FEATURES SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

print(X)
print(y)

#Traning the SVR Model in whole dataset

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)
#Predicting a new result
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))

#Visualizing the results
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = 'red')
plt.title('Truth or Bluff(SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the SVR result(For high resolution and smooth curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)),0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.inverse_transform(X_grid)).reshape(-1,1)),color = 'yellow')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()