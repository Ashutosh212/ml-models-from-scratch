import pandas as pd
import numpy as np
from .model import KMeans

def train_model(data, clusters = 3):
    
    model = KMeans(clusters=clusters)
    return model.fit(data=data)

def wcss(data, cluster_index, centroids):
    '''
    WCSS (Within-Cluster Sum of Squares) measures the sum of squared distances between each point and its assigned cluster centroid
    :param data: data sample
    :param cluster_index: list of assigned clusters to each data point
    :param centroids: centroid co-ordinate of each data point
    '''
    total_wcss = 0.0
    for i in range(len(centroids)):
        cluster_points = data[np.where(cluster_index == i)[0]]
        if len(cluster_points) == 0:
            continue  
        squared_distances = np.sum((cluster_points - centroids[i]) ** 2)
        total_wcss += squared_distances
    return total_wcss
