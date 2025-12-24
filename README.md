# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load Data

2.Split Dataset into Training and Testing Sets

3.Train the Model Using Stochastic Gradient Descent (SGD)

4.Make Predictions and Evaluate Accuracy

5.Generate Confusion Matrix


## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: LOGESHWARAN S
RegisterNumber:25007255  
*/import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_testy_test = train_test_split(X, y, test_size=0.2, random_state=42, y_train, )

sgd_clf = SGDClassifier(max_iter = 1000, tol = 1e-3)
sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

```

## Output:

<img width="874" height="287" alt="Screenshot 2025-12-24 091218" src="https://github.com/user-attachments/assets/3e8737d9-7277-4ecb-b434-0f26354987e1" />


<img width="253" height="107" alt="Screenshot 2025-12-24 091235" src="https://github.com/user-attachments/assets/d2f6f4f0-253f-4448-9cdb-72cf0253fd7d" />


<img width="194" height="38" alt="Screenshot 2025-12-24 091243" src="https://github.com/user-attachments/assets/4d1f1406-88e1-48a9-b0e3-be8bc2d333aa" />


<img width="172" height="110" alt="Screenshot 2025-12-24 091249" src="https://github.com/user-attachments/assets/0bb5c5ce-c428-46e6-a45b-56a5648e183e" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
