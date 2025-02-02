import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Generate synthetic customer data
np.random.seed(42)
num_customers = 200
num_categories = 5
purchase_history = np.random.randint(0, 1000, size=(num_customers, num_categories))
categories = [f'Category_{i+1}' for i in range(num_categories)]
customer_data = pd.DataFrame(purchase_history, columns=categories)

# Step 2: Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data)

# Step 3: Apply K-means clustering
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_data)
customer_data['Cluster'] = kmeans.labels_

# Step 4: Analyze the clusters
cluster_means = customer_data.groupby('Cluster').mean()
print(cluster_means)

# Step 5: Visualize the clusters (Optional)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(10, 6))
for i in range(k):
    plt.scatter(reduced_data[customer_data['Cluster'] == i, 0],
                reduced_data[customer_data['Cluster'] == i, 1],
                label=f'Cluster {i}')
plt.title('Customer Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()