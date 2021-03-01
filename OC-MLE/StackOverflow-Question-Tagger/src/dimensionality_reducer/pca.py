from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import axes3d
from sklearn.decomposition import PCA #, TruncatedSVD

from src.preprocessor import standard_scaler
from src.datacleaner import delete_cols, get_centroids_from_categories
from src.analyzer.univariate import pd, np, plt, percentage_change  # sns; sns.set()
from src.evaluator import elbow_criterion


class LinearPCA:

    def __init__(self, df, reduce=True, n_comp=None, preprocess_data=False, category_label=None):
        if preprocess_data:
            df = df.copy().set_index(category_label)
        df.index = df.index.astype('category')  # ! : if index values are numeric it will increase memory usage
        self.data = df
        self.category_label = df.index.name if category_label is None else category_label
        self.cat_colors = df.index.codes
        self.X = df.values
        # Center & reduce
        self.X_scaled = standard_scaler(self.X, reduce=reduce)
        self.categories = df.index
        self.features = df.columns
        # Get main components
        self.n_comp = self.get_max_components() if n_comp is None else n_comp
        self.pca = PCA(n_components=self.n_comp)
        self.X_projected = ""
        self.components_table = ""
        self.default_factorial_plan_nb = ""
        self.evr = "" # explained variance ratio

########################################################################################################################
#                                           Parameters                                                                 #
########################################################################################################################

    def get_max_components(self):
        """
        Get the number of maximum components
        """
        n, p = self.data.shape
        return min([p, n - 1])

    def kaiser_criterion(self):
        """
        Find total components number based on Kaiser criterion : (100 / p, where p is components number)
        """
        # Only consider axes whose associated inertia is less than 100/p
        total_comp = len([r for r in self.evr if r > (100 / self.n_comp) / 100])
        return total_comp

    def fit(self):
        """

        """
        self.pca.fit(self.X_scaled)
        self.X_projected = self.pca.transform(self.X_scaled)
        self.evr = self.pca.explained_variance_ratio_
        # Components table
        components_cols = ["F{}".format(n + 1) for n in range(self.n_comp)]
        self.components_table = pd.DataFrame(self.X_projected, index=self.categories, columns=components_cols)
        self.default_factorial_plan_nb = int(self.kaiser_criterion() // 2)
        
    def get_centroids(self):
        """

        """
        centroid_df = pd.DataFrame(self.X_projected)
        centroid_df[self.category_label] = self.categories
        df_centroids_projected = get_centroids_from_categories(centroid_df, self.category_label)
        X_centroids_projected = df_centroids_projected.values
        centroids_categories = df_centroids_projected.index
        return X_centroids_projected, centroids_categories

########################################################################################################################
#                                           Visualizations                                                             #
########################################################################################################################

    def scree_plot(self, threshold=None, save_as_img=False):  # (% Explained Variance)
        """

        """
        scree = self.evr * 100
        plt.bar(np.arange(len(scree)) + 1, scree)
        if threshold is not None:
            scree_freq = scree / scree.sum()
            scree_cumsum = np.cumsum(scree_freq)
            # Number of features needed for threshold cumulative importance
            n_features = np.min(np.where(scree_cumsum > threshold)) + 1
            threshold_percentage = 100 * threshold
            threshold_legend = '{} features required for {:.0f}% of inertia.'.format(n_features, 
                                                                                     threshold_percentage)
            # Threshold  vertical line plot
            plt.vlines(n_features, ymin=0, ymax=threshold_percentage, linestyles='--', colors='red')
            plt.plot(np.arange(len(scree)) + 1, scree.cumsum(), c="red", marker='o', label=threshold_legend)
            plt.legend(loc='lower right', fontsize=12)
        else:
            plt.plot(np.arange(len(scree)) + 1, scree.cumsum(), c="red", marker='o')
        plt.xlabel("Inertia axis rank", labelpad=20)
        plt.ylabel("Inertia (%)", labelpad=20)
        plt.title("Scree plot" +
                  "\n(Kaiser criterion = {} : Elbow criterion = {})".format(self.kaiser_criterion(),
                                                                            elbow_criterion(total_inertia=self.evr)),
                                                                            pad=20)
        if save_as_img:
            plt.tight_layout()
            plt.savefig('scree.jpg')
        plt.show(block=False)

    def plot_factorial_planes(self,
                              n_plan=None,
                              X_projected=None,
                              labels=None,
                              alpha=1,
                              illustrative_var=None,
                              illustrative_var_title=None,
                              save_as_img=False,
                              plot_size=(10, 8)):
        """
        :param: axis_nb: the total number of axes to display (default is kaiser criterion divided by 2)
        """
        X_projected = self.X_projected if X_projected is None else X_projected
        factorial_plan_nb = self.default_factorial_plan_nb if n_plan is None else n_plan
        axis_ranks = [(x, x + 1) for x in range(0, factorial_plan_nb, 2)]
        for d1, d2 in axis_ranks:
            if d2 < self.n_comp:
                fig = plt.figure(figsize=plot_size)
                # Display data points
                if illustrative_var is None:
                    plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
                else:
                    illustrative_var = np.array(illustrative_var)
                    for value in np.unique(illustrative_var):
                        selected = np.where(illustrative_var == value)
                        plt.scatter(X_projected[selected, d1],
                                    X_projected[selected, d2], alpha=alpha, label=value)
                    plt.legend(title=illustrative_var_title if illustrative_var_title is not None else None)
                # Display data points labels
                if labels is not None:
                    for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
                        plt.text(x, y, labels[i], fontsize='12', ha='center', va='bottom')
                        # Fix factorial plan limits
                boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
                plt.xlim([-boundary, boundary])
                plt.ylim([-boundary, boundary])
                # Display horizontal & vertical lines
                plt.plot([-100, 100], [0, 0], color='grey', ls='--')
                plt.plot([0, 0], [-100, 100], color='grey', ls='--')
                # Axes labels with % explained variance
                plt.xlabel('F{} ({}%)'.format(d1+1, round(100*self.evr[d1], 1)), labelpad=20)
                plt.ylabel('F{} ({}%)'.format(d2+1, round(100*self.evr[d2], 1)), labelpad=20)
                plt.title("Projection des individus (sur F{} et F{})".format(d1 + 1, d2 + 1), pad=20)
                if save_as_img:
                    plt.tight_layout()
                    plt.savefig('factorial_plan_{}.jpg'.format(1 if d1 == 0 else d1))
                plt.show(block=False)

    def plot_3d_factorial_plan(self, x_comp=1, y_comp=2, z_comp=3, cat_colors=None):
        """

        """
        # Store results of PCA in a data frame
        # result = pd.DataFrame(self.X_projected, columns=['PCA{}'.format(i+1) for i in range(self.n_comp)])
        cat_colors = self.cat_colors if cat_colors is None else cat_colors
        result = self.components_table
        my_dpi = 96
        fig = plt.figure(figsize=(480 / my_dpi, 480 / my_dpi), dpi=my_dpi)  # fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_aspect('equal')
        axes_3d_comp = [x_comp, y_comp, z_comp]
        x_comp_label, y_comp_label, z_comp_label = ['F{}'.format(n) for n in axes_3d_comp]  # old "PCA"
        # Components axes limits
        xmin, xmax = (min(result[x_comp_label]), max(result[x_comp_label]))
        ymin, ymax = (min(result[y_comp_label]), max(result[y_comp_label]))
        zmin, zmax = (min(result[z_comp_label]), max(result[z_comp_label]))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        # Components axes coordinates
        xaxis = [(xmin, xmax), (0, 0), (0, 0)]
        yaxis = [(0, 0), (ymin, ymax), (0, 0)]
        zaxis = [(0, 0), (0, 0), (zmin, zmax)]
        # Plot components axes
        for a in [xaxis, yaxis, zaxis]:
            ax.plot(a[0], a[1], a[2], 'b')
        # label the axes
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        fig.tight_layout()
        # Plot
        ax.scatter(result[x_comp_label], result[y_comp_label], result[z_comp_label], c=cat_colors, cmap="Set2_r", s=60)
        plt.title("3D PCA")
        plt.show()

    def plot_correlation_circle(self,
                                n_plan=None,
                                labels=None,
                                label_rotation=0,
                                lims=None,
                                save_as_img=False,
                                plot_size=(10, 8)):
        """

        """
        factorial_plan_nb = self.default_factorial_plan_nb if n_plan is None else n_plan
        # Build a list of tuples (example : [(0, 1), (2, 3), ... ])
        axis_ranks = [(x, x + 1) for x in range(0, factorial_plan_nb, 2)]
        pcs = self.pca.components_
        for d1, d2 in axis_ranks:
            if d2 < self.n_comp:
                fig, ax = plt.subplots(figsize=plot_size)
                # Fix factorial plan limits
                if lims is not None:
                    xmin, xmax, ymin, ymax = lims
                elif pcs.shape[1] < 30:
                    xmin, xmax, ymin, ymax = -1, 1, -1, 1
                else:
                    xmin, xmax, ymin, ymax = min(pcs[d1, :]), max(pcs[d1, :]), min(pcs[d2, :]), max(pcs[d2, :])
                # affichage des flèches
                # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
                if pcs.shape[1] < 30:
                    plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]), pcs[d1, :], pcs[d2, :],
                               angles='xy', scale_units='xy', scale=1, color="grey")
                    # (doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
                else:
                    lines = [[[0, 0], [x, y]] for x, y in pcs[[d1, d2]].T]
                    ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
                # Display variables labels
                if labels is not None:
                    for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                        if xmin <= x <= xmax and ymin <= y <= ymax:
                            plt.text(x, y, labels[i], fontsize='14', ha='center', va='center',
                                     rotation=label_rotation, color="blue", alpha=0.5)  # fontsize : 14
                # Plot circle
                circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
                plt.gca().add_artist(circle)
                # définition des limites du graphique
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
                # affichage des lignes horizontales et verticales
                plt.plot([-1, 1], [0, 0], color='grey', ls='--')
                plt.plot([0, 0], [-1, 1], color='grey', ls='--')
                # Axes labels with % explained variance
                plt.xlabel('F{} ({}%)'.format(d1+1, round(100*self.evr[d1], 1)), labelpad=20)
                plt.ylabel('F{} ({}%)'.format(d2+1, round(100*self.evr[d2], 1)), labelpad=20)
                plt.title("Cercle des corrélations (F{} et F{})".format(d1 + 1, d2 + 1), pad=20)
                if save_as_img:
                    plt.tight_layout()
                    plt.savefig('corr_circle_{}.jpg'.format(1 if d1 == 0 else d1))
                plt.show(block=False)
