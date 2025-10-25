#Standard imports
from filecmp import cmp
from random import Random
from igraph import palettes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import preprocessing stuff
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

#Import the modeling stuff
from sklearn.ensemble import RandomForestClassifier

#Evaluation stuff
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

#Get the dataset
X, y = fetch_openml(
    name='adult',
    version = 1,
    return_X_y=True
)

#Preprocessing
oe = OrdinalEncoder()
le = LabelEncoder()

#Fit the features
X[X.select_dtypes(include = ['category']).columns] = oe.fit_transform(X[X.select_dtypes(include = ['category']).columns])

y = le.fit_transform(y)

#Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=0.3,
    random_state=42, 
    shuffle=True
)

#Now fit the model
rf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_preds = rf.predict(X_test)

#Now we get the evals
conf = confusion_matrix(
    y_test, 
    y_preds
)

plt.title('Confusion Matrix for Random Forest on Adult Income Dataset')
sns.heatmap(conf, cmap='Blues', annot=True, fmt='d')
plt.show()

#Now we use classification report
classif =  pd.DataFrame(
    classification_report(
    y_test,
    y_preds,
    output_dict=True
))


fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])

plt.plot(fpr, tpr, label='Random Forest')
plt.plot([0,1], [0,1], linestyle='--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest on Adult Income Dataset')
plt.legend()
plt.show()


