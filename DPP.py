import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('C:/Users/Sundram Vaths/Downloads/Stipend_Data.csv')

X=dataset.iloc[:,0:1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_predict=regressor.predict(X_test)

plt.scatter(X_train,y_train,color='Red')
plt.plot(X_train,regressor.predict(X_train),color='Blue')
plt.title('Traning Data Set')
plt.xlabel('years of exp')
plt.ylabel('stipned')

plt.scatter(X_test,y_test,color='Red')
plt.plot(X_train,regressor.predict(X_train),color='Blue')
plt.title('Test Data Set')
plt.xlabel('years of exp')
plt.ylabel('stipned')

from sklearn import metrics
print(metrics.mean_absolute_error(y_test,y_predict))

print(metrics.mean_squared_error(y_test,y_predict))
