import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('D:/Position_Salaries.csv')

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

print(X)
print(y)

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

regressor.predict([[6.5]])
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X , y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth of Bluff (Decision Tree Regressor)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
