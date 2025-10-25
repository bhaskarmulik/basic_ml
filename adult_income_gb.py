from operator import index
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import cast

import sklearn
from sklearn.datasets import fetch_openml
import sklearn.ensemble
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

#fetch data
X, y = fetch_openml(
    name='adult',
    version=1,
    return_X_y=True,
    as_frame=True  # Ensure we get pandas DataFrame/Series
)

# Cast to correct types
X = cast(pd.DataFrame, X)
y = cast(pd.Series, y)

#Encode using label encoder
le = LabelEncoder()
oe = OrdinalEncoder()

X[X.select_dtypes(include = ['category']).columns] = oe.fit_transform( X[X.select_dtypes(include = ['category']).columns])

y = le.fit_transform(y)
y = cast(pd.Series, y)

X.dropna(inplace = True)
# Ensure y has the same index as X
y = y[X.index]


#Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size = 0.3, shuffle=True, random_state=42
)

#Initialize the model
gb1 = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

gb2 = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

ab = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1
)

#Fit to just the graient boosting model
gb1.fit(X_train, y_train)

#Fit to teh adaboost model
ab.fit(X_train, y_train)

#Now for evals

def get_evals(model : sklearn.ensemble.GradientBoostingClassifier | sklearn.ensemble.AdaBoostClassifier) -> None:

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)
    pd.DataFrame(y_pred_prob).head()
    y_pred_prob = y_pred_prob[:,1]

    #Confusion matrix
    conf = confusion_matrix(y_test, y_pred)

    #AUC-ROC curve
    tpr, fpr, thresh = roc_curve(
        y_test,
        y_pred_prob
    )

    #Feature imp
    feature_imp = model.feature_importances_
    feat_imp = pd.Series(feature_imp, index=X.columns).sort_values(ascending=False)

    fig, ax = plt.subplots(
        nrows=2,
        ncols = 2,
        figsize=(30,10)
    )

    #precision recall curve
    pres, recall, threshold = precision_recall_curve(
        y_test, 
        y_pred_prob
    )

    ax = ax.flatten()

    sns.heatmap(conf, ax=ax[0], annot=True, fmt='d')
    sns.lineplot(
        x = tpr, 
        y = fpr,
        palette='rocket',
        ax=ax[1]
    )
    ax[1].plot([0,1], [0,1], color='k', linestyle='--')

    sns.barplot(x=feat_imp.index, y=feat_imp.values, width=0.5, ax=ax[2])
    sns.lineplot(
        x = recall,
        y=pres,
        palette='coolwarm',
        ax=ax[3]
    )
    ax[3].plot([0,1], [1,0], "k--")
    plt.show()
    pass

get_evals(gb1)
get_evals(ab)
