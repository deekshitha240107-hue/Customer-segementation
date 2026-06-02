import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
df = pd.read_csv('C:/Users/24331/customer_data.csv')

print(df.head())
print(df.info())
print(df.isnull().sum())
df = df.dropna()
#age disturbution
plt.hist(df['Age'], bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
#spending score disturbution
plt.hist(df['SpendingScore'], bins=10)
plt.title("Spending Score Distribution")
plt.xlabel("Score")
plt.ylabel("Count")
plt.show()
X = df[['Income', 'SpendingScore']]
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
wcss = []

for i in range(1,11):
    kmeans = KMeans(
        n_clusters=i,
        random_state=42
    )

    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()
kmeans = KMeans(
    n_clusters=4,
    random_state=42
)

df['Cluster'] = kmeans.fit_predict(X_scaled)
print(df.head())
plt.figure(figsize=(8,6))

sns.scatterplot(
    x='Income',
    y='SpendingScore',
    hue='Cluster',
    data=df,
    palette='Set1'
)

plt.title("Customer Segmentation")
plt.show()
segment_summary = df.groupby('Cluster')[
    ['Income','SpendingScore','PurchaseFrequency']
].mean()

print(segment_summary)


