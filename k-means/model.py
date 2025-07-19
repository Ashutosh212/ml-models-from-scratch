import numpy as np
import sys
import matplotlib.pyplot as plt
np.random.seed(42)

sys.path.append('/home/ashu/Projects/ml-models-from-scratch')

from toolkit.distances import SquaredEuclideanDistance

class KMeans:
    def __init__(self, clusters):
        self.clusters = clusters
    
    def fit(self, data):
        self.n = data.shape[1]
        self.sample = data.shape[0]
        initial_indices = np.random.choice(len(data), self.clusters, replace=False)

        self.centroids = data[initial_indices]
        self.cluster_index = np.zeros(self.sample)
        # epoch = 1
        while True:
            reassigned = False
            # print(f"Epoch: {epoch}")
            # epoch += 1

            for i, sample in enumerate(data[:]):
                curr_cluster = np.argmin([SquaredEuclideanDistance(sample, cluster) for cluster in self.centroids])
                if self.cluster_index[i] != curr_cluster:
                    reassigned = True
                
                self.cluster_index[i] = curr_cluster
            
            if not reassigned:
                break
        
            for i in range(self.clusters):
                cluster_points = data[np.where(self.cluster_index == i)[0]]
                # print(f"Number of data points associated with cluster {i} is {len(cluster_points)}")


                if len(cluster_points) == 0:
                    # Reinitialize this centroid to a random data point
                    self.centroids[i] = data[np.random.choice(len(data))]
                    print(f"Reinitialized empty cluster {i}")
                else:
                    self.centroids[i] = np.mean(cluster_points, axis=0)  
        return self.cluster_index, self.centroids
    
def plot_clusters(data, cluster_index, centroids=None):
    plt.figure(figsize=(8, 6))
    k = int(np.max(cluster_index)) + 1
    colors = plt.cm.get_cmap("tab10", k)

    for i in range(k):
        cluster_data = data[cluster_index == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=40, color=colors(i), label=f'Cluster {i}')
    
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], s=200, color='black', marker='X', label='Centroids')
   
    plt.savefig("sample_data_clusters.png")


def main():
    data_points = np.random.randint(1,10, (20,2))
    k = 3
    model = KMeans(k)
    cluster_index, centroids = model.fit(data_points)
    plot_clusters(data_points, cluster_index, centroids)
    print(cluster_index)

if __name__=="__main__":
    main()