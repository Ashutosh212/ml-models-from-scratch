import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .train import train_model, wcss


df = pd.read_csv("k-means/dataset/Mall_Customers.csv")

# data pre-processing
df = df.drop('CustomerID', axis=1)
df['Gender'] = np.where(df['Gender']=="Male", 0, 1)

# print(df.head())

data = np.array(df)
# print(data.shape)

wcss_list = []
for clusters in range(1, 9):
    cluster_index, centroids = train_model(data, clusters=clusters)
    cluster_wcss = wcss(data, cluster_index, centroids)
    wcss_list.append(cluster_wcss)

print(wcss_list)

def save_wcss_plot(wcss_list, filename="wcss_vs_clusters.png"):
    clusters_range = range(1, len(wcss_list) + 1)
    plt.plot(clusters_range, wcss_list, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()

save_wcss_plot(wcss_list)