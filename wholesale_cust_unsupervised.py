import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#get the dataset
from sklearn.datasets import fetch_openml

data = fetch_openml(
    name = 'wholesale-customers',
    version="active",
    as_frame=True
)

df = data.frame

#Inspect the dataset
df.info()

df = df.select_dtypes(exclude = 'category')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

df = pd.DataFrame(
    scaled_df,
    columns=df.columns
)

df.describe()

wcss = list()
for i in range(1, 25):
    km = KMeans(n_clusters=i, n_init=10, max_iter = 300)

    km.fit(df)
    wcss.append(km.inertia_)


plt.plot(
    range(1,25),
    wcss,
    marker="o",
    linestyle = "--" 
)

plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

optimal_pt = 4

kmeans = KMeans(
    n_clusters=optimal_pt,
)

clusters = kmeans.fit_predict(df)

df["clusters"] = clusters

df.clusters.value_counts()

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df.drop("clusters", axis=1))

df["pca-one"] = pca_result[:,0]
df["pca-two"] = pca_result[:,1]

plt.figure(figsize=(10,8))
sns.scatterplot(
    x="pca-one",
    y="pca-two",
    hue="clusters",
    palette=sns.color_palette("hsv", as_cmap=True),
    data=df,
    legend="full",
    alpha=0.7
)

plt.title("KMeans Clustering of Wholesale Customers Dataset")
plt.show()
