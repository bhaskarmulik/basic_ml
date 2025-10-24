import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#fetch the titanic dataset

data = fetch_openml(
    name = 'titanic',
    as_frame = True,
    version = 1
)

df = data.data
target = data.target

df.head()

# target.head()

df.info()
df.isnull().sum()

#Remove the rows like cabin, boat, body and home.dest
cols_to_remove = df.isnull().sum()[df.isnull().sum() > 500].index.tolist()
cols_to_remove.extend(['name', 'ticket'])
df.drop(cols_to_remove, inplace=True, axis=1)

sns.pairplot(df)

df.dropna(inplace=True)
y = target.loc[df.index]

#Now we build a Logistic Regression model

df = pd.get_dummies(df, drop_first=True)
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)


