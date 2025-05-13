# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Aman Singh
RegisterNumber:  212224040020
*/
```
```python
import pandas as pd
df=pd.read_csv("/content/Employee.csv")
print("data.head():")
df.head()

print("data.info()")
df.info()


print("data.isnull().sum()")
df.isnull().sum()

print("data value counts")
df["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
df["salary"]=le.fit_transform(df["salary"])
df.head()

print("x.head():")
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=df["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred) # dt.score(X_test,y_test)
accuracy

print("Data prediction")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plot_tree(dt,filled=True,feature_names=x.columns,class_names=['Stayed' , 'Left'])
plt.show()
```

## Output:

<img width="953" alt="Screenshot 2025-05-13 at 10 34 40 AM" src="https://github.com/user-attachments/assets/c126d9e9-d44c-4957-9777-cdc457165d46" />


<img width="484" alt="Screenshot 2025-05-13 at 10 34 54 AM" src="https://github.com/user-attachments/assets/7a1dc807-5ea2-4600-b884-728369ae8d4e" />


<img width="317" alt="Screenshot 2025-05-13 at 10 35 09 AM" src="https://github.com/user-attachments/assets/8ff10817-6d0f-45ae-a8dc-e9a19849dc03" />


<img width="313" alt="Screenshot 2025-05-13 at 10 35 21 AM" src="https://github.com/user-attachments/assets/753ac6ef-e0b2-41ab-b92b-c18dc15355b9" />


<img width="936" alt="Screenshot 2025-05-13 at 10 35 56 AM" src="https://github.com/user-attachments/assets/fdf93cbc-2b2f-424b-896b-1bde1c58318d" />


<img width="974" alt="Screenshot 2025-05-13 at 10 35 40 AM" src="https://github.com/user-attachments/assets/cdd1945e-12cd-4ccc-9919-4698dcf425b8" />

<img width="945" alt="Screenshot 2025-05-13 at 10 36 32 AM" src="https://github.com/user-attachments/assets/dbfa204f-a2a8-40fb-9daa-69282db94f1a" />

<img width="935" alt="Screenshot 2025-05-13 at 10 36 54 AM" src="https://github.com/user-attachments/assets/59bef747-ba22-4403-921d-2644c6f93cd5" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
