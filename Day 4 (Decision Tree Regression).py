#Lib...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

#Importing Dataset 
dataset = pd.read_csv('D:/Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
 
print(X)
print(y.reshape(-1,1))

#Traning the Decision Tree regression

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#predict the new result
regressor.predict([[6.5]])
plt.show()
