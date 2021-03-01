from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from src.analyzer.univariate import np, pd, plt, sns
sns.set_style("whitegrid", {'axes.grid' : False})


class tSNE:
    def __init__(self, df, n_comp=2, perplexity=30, init='random', verbose=False):
        self.data = df
        self.n_comp = n_comp
        self.perplexity = perplexity
        self.init = init
        self.verbose = verbose
        self.X_embedded = ""
        self.df_embedded = ""
        
    def fit(self):
        self.X_embedded = TSNE(n_components=self.n_comp,
                               perplexity=self.perplexity,
                               verbose=self.verbose,
                               init=self.init).fit_transform(self.data)
        self.df_embedded = pd.DataFrame({f'{n+1}d': self.X_embedded[:, n] for n in range(self.n_comp)})
        
    def plot(self, plot_size=(16, 10), cluster_labels=None, c=10):
        if cluster_labels is not None:
            cluster_title = cluster_labels.name.capitalize()
            self.df_embedded[cluster_title] = cluster_labels
            n_colors = len(self.df_embedded[cluster_title].unique())
            centroids_series = [self.df_embedded[self.df_embedded[cluster_title] == n+1].mean() for n in range(n_colors)]
            centroids_df = pd.concat(centroids_series, axis=1).T
        # 2D plot
        if self.n_comp == 2:
            plt.figure(figsize=plot_size)
            # Plot clusters
            ax1 = sns.scatterplot(x="1d", y="2d", hue=cluster_title if cluster_labels is not None else None,
                                 palette=sns.color_palette("hls", n_colors if cluster_labels is not None else c),
                                 data=self.df_embedded,
                                 legend="full",
                                 alpha=0.3)
            if cluster_labels is not None:
                # Plot centroids
                for i in range(n_colors):
                    ax1.scatter(centroids_df.iloc[i, 0],
                                centroids_df.iloc[i, 1],
                                c='b',
                                s=50,
                                ec='black'
                                #label='centroid'
                                )
        # 3D plot
        elif self.n_comp == 3:
            #if cluster_labels[cluster_title.lower()].dtypes.name in ['category', 'object']:
            self.df_embedded[cluster_title] = [i+1 for i in range(len(cluster_labels))]
            fig = plt.figure(figsize=plot_size)
            ax = Axes3D(fig)
            ax.scatter(self.df_embedded.iloc[:, 0],
                       self.df_embedded.iloc[:, 1],
                       self.df_embedded.iloc[:, 2],
                       c=self.df_embedded[cluster_title].values if cluster_labels is not None else 'b',
                       cmap='viridis' if cluster_labels is not None else None,
                       marker='o')
            #ax.legend()
            if cluster_labels is not None:
                # Plot centroids
                for i in range(n_colors):
                    ax.scatter(centroids_df.iloc[i, 0],
                               centroids_df.iloc[i, 1],
                               centroids_df.iloc[i, 2],
                               c='r',
                               s=50,
                               #label='centroid'
                               )
                        #ax.legend()
        else:
            raise Exception()