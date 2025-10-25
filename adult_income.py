#Importing standard libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Import scikit-learn stuff
from sklearn.datasets import fetch_openml

#Preprocessing stuff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

#Modeling 
from sklearn.tree import DecisionTreeClassifier

#Evaluation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import plot_tree




#fetch dataset from openml
X, y = fetch_openml(
    name='adult',
    return_X_y=True,
    version=1
)

#Checkout the dataset
X.head()
y.head()       #Ignore
X.info()

#Check the unique values in each
X.select_dtypes(include = 'category').nunique()


oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_encoded = X.copy()
X_encoded[X_encoded.select_dtypes(include = 'category').columns] = oe.fit_transform(X_encoded.select_dtypes(include = 'category'))
X_encoded.head()


le = LabelEncoder()
y_enc = le.fit_transform(y)




X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, 
    y_enc, 
    test_size=0.3, 
    random_state=42
    )
desc_tree = DecisionTreeClassifier()

desc_tree.fit(X_train, y_train)

y_hat = desc_tree.predict(X_test)


classification_report(
    y_test,
    y_hat
)

confusion_matrix(y_test, y_pred=y_hat)


plt.figure(figsize=(20, 10)) 
plot_tree(
    desc_tree, class_names=y.value_counts().index,
    max_depth=3, feature_names=X.columns,
    filled=True, rounded=True

)
plt.show()

