from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import KMeans
from src.analyzer.descriptor.univar import np, plt
from src.datacleaner.preprocessor import standard_scaler


########################################################################################################################
#                                               Clustering functions                                                   #
########################################################################################################################


# K-means
def kmeans(df, cluster_number):
    x = df.values
    x_scaled = standard_scaler(x)
    # Clustering par K-means
    km = KMeans(n_clusters=cluster_number)
    km.fit(x_scaled)
    clusters = km.labels_
    centroids = km.cluster_centers_
    return clusters, centroids


# Hierarchical clustering with scipy
def hierarchical_clustering(df, method='ward', metric='euclidean', optimal_ordering=False, scale=True):
    x = df.values
    categories = df.index
    if scale:
        # Reduce & center
        x_scaled = standard_scaler(x)
    # Hierarchical clustering
    Z = linkage(x_scaled if scale is True else x, method, metric, optimal_ordering)
    return Z, categories


# Plot dendrogram
def plot_hierarchical_clustering(Z, categories, plot_size=(10, 25), save_as_img=False):
    plt.figure(figsize=plot_size)
    plt.title('Hierarchical Clustering Dendrogram', pad=20)
    plt.xlabel('distance', labelpad=20)
    result_dict = dendrogram(Z, labels=categories, orientation="left", leaf_font_size=16)
    if save_as_img:
        plt.tight_layout()
        plt.savefig('cah.jpg')
    plt.show()


def get_n_clusters_from_dendrogram(df, Z, n, crit='maxclust'):  # TO DO => Generalize for each clustering algorithm
    n_clusters = fcluster(Z, n, criterion=crit)
    clusters_data = []
    for n in range(1, max(n_clusters) + 1):
        # Get subcluster indexes from dendrogram metacluster considered
        sub_cluster_ind = np.where(n_clusters == n)[0]
        # Get cluster indexes from former hierarchical clustering input data frame
        cluster_ind = np.where([i in sub_cluster_ind for i in range(len(df.index))])
        # Build cluster dataframe from cluster indexes
        cluster_df = df.iloc[cluster_ind]
        clusters_data.append(cluster_df)
    return clusters_data
