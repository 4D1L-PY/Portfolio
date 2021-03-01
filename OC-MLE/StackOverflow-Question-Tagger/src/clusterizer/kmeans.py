from src.analyzer.univariate import np, pd, plt
from src.preprocessor import standard_scaler
from src.evaluator import elbow_criterion
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from yellowbrick.cluster.elbow import kelbow_visualizer # warning sklearn.metrics old version


class Kmeans:

    def __init__(self, df, k_range=None, max_iter=300, random_state=42, parallelize=None):
        # Default attributes
        self.data = df
        self.n_clusters = k_range
        self.max_iter = max_iter
        self.random_state = random_state
        self.parallelize = parallelize
        # k-means attributes
        self.X = self.data.values
        self.categories = df.index
        self.category_label = self.categories.name
        self.features = df.columns
        # k-means output data
        self.model = ''
        self.clusters = ''
        self.centroids = ''
        self.inertia = ''
        self.optimal_clusters_nb = ''
        self.silhouette_score = ''
        self.davies_bouldin_score = ''

    def fit(self):
        self.model = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, random_state=self.random_state)
        self.model.fit(self.X)
        self.clusters = self.model.labels_
        self.centroids = self.model.cluster_centers_
        self.inertia = self.model.inertia_
        # Evaluation
        # self.silhouette_score = silhouette_score(self.X, self.clusters, n_jobs=self.parallelize)
        self.davies_bouldin_score = davies_bouldin_score(self.X, self.clusters)
        
    def reduce_cluster_data(self, n=50):
        reduced_chunks = []
        k_clusters = range(self.n_clusters)
        for k in k_clusters:
            distances = self.model.transform(self.X)[:, k]
            data_idx = np.argsort(distances)[::-1][:n]
            reduced_chunk = self.data.iloc[data_idx]
            reduced_chunk.loc[:, 'clusters'] = k+1
            reduced_chunks.append(reduced_chunk)
        reduced_data = pd.concat(reduced_chunks)
        return reduced_data

    def plot_kelbow_visualizer(self, scorer='distortion', axis=None, plt_show=True):
        return kelbow_visualizer(KMeans(), self.data, k=self.n_clusters, metric=scorer, ax=axis, show=plt_show)
    
#     def plot_elbow_method(self, n_clusters,
#                           thr=0.25,
#                           plot_size=None,
#                           title_pad=20,
#                           xlabel_pad=20,
#                           ylabel_pad=20,
#                           return_data=False,
#                           save_as_img=False,
#                           filename='scree',
#                           file_type='jpg'):
#         cluster_inertia = []
#         K = range(1, n_clusters+1)
#         for k in K:
#             model = KMeans(n_clusters=k, max_iter=self.max_iter, random_state=self.random_state)
#             model.fit(self.X)
#             cluster_inertia.append(model.inertia_)
#         # Compute elbow criterion
#         self.optimal_clusters_nb = elbow_criterion(total_inertia=np.array(cluster_inertia), threshold=thr)
#         if plot_size:
#             plt.figure(figsize=plot_size)
#         plt.plot(K, cluster_inertia, 'bx-', label=f'{self.optimal_clusters_nb} optimal clusters')
#         plt.vlines(self.optimal_clusters_nb,
#                    ymin=0,
#                    ymax=cluster_inertia[self.optimal_clusters_nb-1],
#                    linestyles='--',
#                    colors='red')
#         plt.xlabel('k', labelpad=xlabel_pad)
#         plt.ylabel('Sum of squared distances (inertia) ', labelpad=ylabel_pad)
#         plt.title(f'Elbow Method For Optimal k From K = {n_clusters}', pad=title_pad)
#         plt.legend(loc='upper right', fontsize=12)
#         if save_as_img:
#             plt.tight_layout()
#             plt.savefig(f'{filename}.{file_type}')
#         plt.show()
#         if return_data:
#             return cluster_inertia
