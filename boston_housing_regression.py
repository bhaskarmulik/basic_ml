import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Fetch the dataset
from sklearn.datasets import fetch_openml

df = fetch_openml(
    name = 'boston', 
    version = 1,
    as_frame = True
)

#Get the target and the data
target = df.target
df = df.data

df.info()

#Lets check for missing values
df.isnull().sum()

#Now lets check the cardinality of the categorical features
cat_df = df.select_dtypes(include = 'category')
print("Number of unique vals in cat cols : \n", cat_df.nunique())
print("Also print the values in each case")
cat_df.value_counts()

#Now let's one hot encode the categorical columns
df = pd.get_dummies(
    df, 
    columns=cat_df.columns.to_list(),
    drop_first=True
)

#Recheck the dataset
df.head()

#Now train a linear regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(df, target, test_size = 0.2, random_state=42)

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print(mean_squared_error(y_pred, y_test))

sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()



