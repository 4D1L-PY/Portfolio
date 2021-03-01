from src.analyzer.univariate import np, plt
from src.preprocessor import standard_scaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


class CAH:

    def __init__(self, df, method='ward',  metric='euclidean', optimal_order=False, scaler=True, reduce=True):
        # Default attributes
        self.data = df
        self.method = method
        self.metric = metric
        self.optimal_ordering = optimal_order
        self.standardize = scaler
        self.reduce = reduce
        # CAH attributes
        self.X = self.data.values
        if self.standardize:  # Center & reduce
            self.X_scaled = standard_scaler(self.X, reduce=reduce)
        self.categories = df.index
        self.category_label = self.categories.name
        self.features = df.columns
        self.Z = ''
        # Dendrogram data
        self.dendrogram_data = ''

    def fit(self):
        x = self.X_scaled if self.standardize is True else self.X
        self.Z = linkage(x, self.method, self.metric, self.optimal_ordering)

    def get_n_clusters(self, n, crit='maxclust'):  # TODO => Generalize for each clustering algorithm
        n_clusters = fcluster(self.Z, n, criterion=crit)
        clusters_data = []
        for n in range(1, max(n_clusters) + 1):
            # Get subcluster indexes from CAH metacluster considered
            sub_cluster_ind = np.where(n_clusters == n)[0]
            # Get cluster indexes from former hierarchical clustering input data frame
            cluster_ind = np.where([i in sub_cluster_ind for i in range(len(self.data.index))])
            # Build cluster dataframe from cluster indexes
            cluster_df = self.data.iloc[cluster_ind]
            clusters_data.append(cluster_df)
        return clusters_data

    def plot_dendrogram(self, plot_size=(10, 25), title_pad=20, xlabel_pad=20, orient='left', leaf_text_size=16,
                        save_as_img=False, filename='cah', file_type='jpg'):
        plt.figure(figsize=plot_size)
        plt.title('Hierarchical Clustering Dendrogram', pad=title_pad)
        plt.xlabel('distance', labelpad=xlabel_pad)
        self.dendrogram_data = dendrogram(self.Z,
                                          labels=self.categories,
                                          orientation=orient,
                                          leaf_font_size=leaf_text_size)
        if save_as_img:
            plt.tight_layout()
            plt.savefig(f'{filename}.{file_type}')
        plt.show()
