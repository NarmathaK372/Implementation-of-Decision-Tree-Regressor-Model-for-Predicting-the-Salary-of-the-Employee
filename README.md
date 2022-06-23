# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Gather information and presence of null in the dataset.
4. From sklearn.tree import DecisionTreeRegressor and fir the model.
5. Find the mean square error and r squared score value of the model.
6. Check the trained model.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: NARMATHA K
RegisterNumber: 212219040088 
*/
```
```
#import packages
import pandas as pd
df=pd.read_csv("Salary.csv")
df.head()

#checking the data information and null presence
df.info()
df.isnull().sum()

#encoding categorical features to numeric
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"] = le.fit_transform(df["Position"])
df.head()

#assigning x and y 
x = df[["Position","Level"]]
y = df["Salary"]

#splitting data into training and test
#implementing decision tree regressor in training model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

#calculating mean square error
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

#calculating r square value
r2 = metrics.r2_score(y_test,y_pred)
r2

#testing the model
dt.predict([[5,6]])
```

## Output:

Initial Dataset:

![ml6](https://user-images.githubusercontent.com/93427345/174307496-2397613b-3709-4f8c-a752-0241cf289038.PNG)

Dataset information:

![ml61](https://user-images.githubusercontent.com/93427345/174307581-de92f788-742b-4aa3-8698-7c5451a51a6d.PNG)

Encoded Dataset:

![ml62](https://user-images.githubusercontent.com/93427345/174307620-f4ac6592-8060-4ffb-9e47-04fc0074e66d.PNG)

Mean Square Error value:

![ml63](https://user-images.githubusercontent.com/93427345/174307645-cbeb6dc8-07f8-44a6-8d9b-c81eb638f84f.PNG)

R squared score:

![ml64](https://user-images.githubusercontent.com/93427345/174307680-380fb4e5-a89b-46ec-8634-062a5cb61785.PNG)

Result value of Model when tested:

![ml65](https://user-images.githubusercontent.com/93427345/174307708-4a637344-05dc-4f10-aa9b-7589ef1093fd.PNG)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
