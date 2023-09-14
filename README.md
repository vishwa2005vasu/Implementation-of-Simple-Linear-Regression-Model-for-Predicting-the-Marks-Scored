# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/student_scores.csv')
#displaying the content in datafile
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
y_pred
#graph plot for training data
plt.scatter(x_train,y_train,color="darkseagreen")
plt.plot(x_train,regressor.predict(x_train),color="plum")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="darkblue")
plt.plot(x_test,regressor.predict(x_test),color="plum")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:
![ml 1](https://github.com/vishwa2005vasu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135954202/fbdd0d9e-5c46-4f21-90f8-d47e5f7d2b2e)
![ml 2](https://github.com/vishwa2005vasu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135954202/97226061-f97c-468f-ac14-b839f0c828df)
![ml 3 - Copy (2)](https://github.com/vishwa2005vasu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135954202/f4469c3b-c33d-416e-bfcf-49a07d85b9df)
![ml 4 - Copy (2)](https://github.com/vishwa2005vasu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135954202/2485495c-b1bb-4db2-b137-ba45af1645fa)
![ml 5 - Copy (2)](https://github.com/vishwa2005vasu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135954202/d5e24472-632f-4f3b-84e4-1517446d79d7)
![ml 6 - Copy (2)](https://github.com/vishwa2005vasu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135954202/dc26335b-d592-4b2a-ba32-83248a67dec8)
![ml 7 - Copy (2)](https://github.com/vishwa2005vasu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135954202/00cc8040-c10b-4bcc-9841-31ad5490b063)
![ml8 - Copy (2)](https://github.com/vishwa2005vasu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135954202/98818659-b4cf-446c-9110-26e3430d2152)
![ml 9 - Copy](https://github.com/vishwa2005vasu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135954202/1933e81a-e68f-4253-8d84-558d990e8b29)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
