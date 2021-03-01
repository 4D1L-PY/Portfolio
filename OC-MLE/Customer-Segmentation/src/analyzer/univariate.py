from src.datacleaner import pd, np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi, sqrt
from pandas_profiling import ProfileReport


########################################################################################################################
#                                            Univariate Analysis                                                       #
########################################################################################################################


def data_scan(df, to_html=False):
    """
    doc : https://github.com/pandas-profiling/pandas-profiling
    """
    data_report = ProfileReport(df, title='Pandas Profiling Report', html={'style': {'full_width': True}})
    if to_html:
        data_report.to_file(output_file="data_scan.html")
    else:
        return data_report


# Measures


def percentage_change(old_value, new_value, precision=2):
    return round(((new_value - old_value)/old_value)*100, precision)


def percentage_diff(v1, v2, precision=2):
    return round(abs((v1 - v2)/((v1 + v2)/2))*100, precision)


def measures_table(df, variable=None, include=None):
    if variable is not None:
        return pd.DataFrame(df[variable].describe(include))
    else:
        return pd.DataFrame(df.describe(include))


def univariate_analysis(df, variable, measure_type='c', measure='mean'):
    if measure_type is 'c':
        return central_tendency_measure(df, variable, measure)
    elif measure_type is 'd':
        return dispersion_measure(df, variable, measure)
    elif measure_type is 's':
        return shape_measure(df, variable, measure)


def central_tendency_measure(df, variable, measure='mean'):
    if type(variable) is str:
        variable = [variable]
    if type(variable) is list:
        for var in variable:
            if measure is "mean":
                measure_value = df[var].mean()
            elif measure is "median":
                measure_value = df[var].median()
            elif measure is "mode":
                measure_value = df[var].mode()
            else:
                print("Error: measure must be mean, median, or mode")
            print("{0} {1} is {2}".format(var, measure, measure_value))
    else:
        print("Error: variable type must be str or list")
    return


def dispersion_measure(df, variable, measure='var'):
    if type(variable) is str:
        variable = [variable]
    if type(variable) is list:
        for var in variable:
            if measure is "var":
                measure_value = df[var].var()
            elif measure is "std":
                measure_value = df[var].std()
            elif measure is "cv":
                measure_value = df[var].std() / df[var].mean()
            elif measure is 'gini':  # ("mesure de concentration" in french)
                # Build Lorenz curve data
                dep = df[variable].values
                n = len(dep)
                lorenz = np.cumsum(np.sort(dep)) / dep.sum()
                lorenz = np.append([0], lorenz)
                plt.axes().axis('equal')
                xaxis = np.linspace(0, 1, len(lorenz))
                # Compute gini index
                AUC = (lorenz.sum() - lorenz[-1] / 2 - lorenz[0] / 2) / n
                S = 0.5 - AUC
                gini = 2 * S
                # Plot Lorenz curve
                plt.plot(xaxis, lorenz, drawstyle='steps-post', label='Gini = {}'.format(round(gini, 2)))
                plt.legend(loc="upper left")
                plt.show()
                return
            else:
                print("Error: measure must be var, std, or cv")
            print("{0} {1} is {2}".format(var, measure, round(measure_value, 2)))
    else:
        print("Error: variable type must be str or list")
    return


def shape_measure(df, variable, measure='skew'):
    if type(variable) is str:
        variable = [variable]
    if type(variable) is list:
        for var in variable:
            if measure is "skew":
                measure_value = df[var].skew()
            elif measure is "kur":
                measure_value = df[var].kurtosis()
            else:
                print("Error: measure must be skew, or kurtosis")
        print("{0} {1} is {2}".format(var, measure, round(measure_value, 2)))
    else:
        print("Error: variable type must be str or list")
    return


# Plot charts


def plot_n_top_values(df, col, n=10, y_label=None, plot_size=(12, 4), annotation_width=None, annotation_height=None):
    """
    """
    grouped_df = df.groupby(col)[col].count().sort_values(ascending=False)
    ax = grouped_df.head(n).plot(kind='barh', figsize=plot_size)
    total = grouped_df.sum()
    annotation_width = (grouped_df.max() / 250) if annotation_width is None else annotation_width
    annotation_height = 1.5 if annotation_height is None else annotation_height
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width() / total)
        x = p.get_x() + p.get_width() + annotation_width # 35
        y = p.get_y() + p.get_height() / annotation_height # 1.5
        ax.annotate(percentage, (x, y), size=12)
    col_label = ' '.join(col.split('_')).capitalize() if '_' in col else col
    plt.gca().invert_yaxis()
    plt.title(f'Top {n} {col_label}', pad=20)
    plt.xlabel('Effectif', labelpad=20)
    plt.ylabel(col_label if y_label is None else y_label, labelpad=20)
    plt.show()
    
    
def count_plot(df, variable, xlabel='Effectif', ylabel=None, plotsize=(15, 8), sort=False, save_as_img=False, n=None):
    if sort:
        df.sort_values(by=[variable], inplace=True)
    plt.figure(figsize=plotsize)
    ax = sns.countplot(y=variable, data=df[:n])
    ylabel = variable.replace('_', ' ') if ylabel is None else ylabel
    plt.title('Distribution of {}'.format(ylabel), pad=20)
    plt.xlabel(xlabel, labelpad=20)
    plt.ylabel(ylabel, labelpad=20)
    total = len(df[variable])
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width() / total)
        x = p.get_x() + p.get_width() + 35
        y = p.get_y() + p.get_height() / 1.5
        ax.annotate(percentage, (x, y), size=12)
    if save_as_img:
        plt.tight_layout()
        plt.savefig('{}_ct_barplot.jpg'.format(variable))
    plt.show()


def plot_empirical_distribution(variable, var_type='quanti', quanti_chart_type='hist', quali_chart_type='pie',
                                density=True, save_as_img=False, xlabel=None, ylabel=None, title=None):
    """
    """
    bar_chart_quanti_cond = (var_type is 'quanti' and quanti_chart_type is 'bar')
    bar_chart_quali_cond = (var_type is 'quali' and quali_chart_type is 'bar')

    if var_type is 'quanti' and quanti_chart_type is 'hist':
        ax = variable.hist(density=density)  # histogramme car nos variables sont continues
        ax.set_xlabel("{}".format(variable.name) if xlabel is None else xlabel, labelpad=20)
        ax.set_ylabel("Effectif" if ylabel is None else ylabel, labelpad=20)
    elif var_type is 'quali' and quali_chart_type is 'pie':
        data = variable.value_counts(normalize=True).sort_index()
        plt.pie(data, labels=[x.upper() if type(x) is str else x.name for x in data.index.tolist()], autopct='%1.1f%%')
        plt.axis('equal')
    elif bar_chart_quali_cond or bar_chart_quanti_cond:
        variable.value_counts(normalize=True).plot(kind='bar', width=0.1 if var_type is 'quanti' else 0.8)
    else:
        if var_type not in ['quali', 'quanti']:
            raise Exception(var_type)
        else:
            raise Exception(var_type)
            
    plt.title("{} distribution".format(variable.name if title is None else title), pad=20)
    if save_as_img:
        plt.tight_layout()
        plt.savefig('{}_dist.jpg'.format(variable.name if title is None else title))
    plt.show()

def plot_dispersion(df, X, Y=None, grouped=True, xlabel='Quantity', ylabel='Variable', save_as_img=False):
    if grouped:
        if type(X) is list:
            uniques_cat = X
        else:
            df.sort_values(by=[X], inplace=True)
            uniques_cat = df[X].unique()
        if Y is None:
            data = [df[df[X] == cat] for cat in uniques_cat]
        else:
            data = [df[df[X] == cat][Y] for cat in uniques_cat]
    else:
        data = df[X]
    # Median & mean graphic properties
    median_props = {'color': "black"}
    mean_props = {'marker': 'o', 'markeredgecolor': 'black', 'markerfacecolor': 'firebrick'}
    plt.boxplot(data, labels=uniques_cat if grouped is True else None, showfliers=False,
                medianprops=median_props, meanprops=mean_props, showmeans=True, vert=False,
                patch_artist=True)
    plt.xlabel(xlabel, labelpad=20)
    plt.ylabel(ylabel, labelpad=20)
    plt.title("{} dispersion".format(ylabel), pad=20)
    if save_as_img:
        plt.tight_layout()
        plt.savefig('{}_boxplot.jpg'.format(X))
    plt.show()


def plot_facet_grid(df, X, Y, Z=None, chart_type='hist', hue=None, hue_order=None, col_wrap=None, h=1,
                    sort_by='x', save_as_img=False):
    df.sort_values(by=[X if sort_by is 'x' else Y], inplace=True)
    g = sns.FacetGrid(df, col=Y, hue=hue, col_wrap=col_wrap, height=h)
    if chart_type is 'hist':
        g = g.map(plt.hist, X)
    elif chart_type is 'scatter':
        g = (g.map(plt.scatter, X, Z)).add_legend()
    if save_as_img:
        plt.tight_layout()
        plt.savefig('{}_facet_grid.jpg'.format(X))
    plt.show()


def plot_radars(df, category_label, save_as_img=False):
    df = df.groupby(category_label).sum()
    df = df.apply(lambda x: round((x / x.sum()) * 100, 1), axis=1)
    max_value = df.values.max()
    df.reset_index(level=0, inplace=True)

    # Part 1 : Build radar chart
    def plot_radar(row, title, color):
        # number of variable
        categories = list(df)[1:]
        N = len(categories)
        # Compute the angle of each axis in the plot (divide the plot by the number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        # Initialise the radar plot
        nrows = ncols = round(sqrt(df.shape[0]))
        ax = plt.subplot(nrows, ncols, row + 1, polar=True)
        # Set the first axis on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        # Draw one axe per variable + add labels
        plt.xticks(angles[:-1], categories, color='grey', size=20)
        # Draw ylabels
        ax.set_rlabel_position(0)
        yticks_values = [int(n) for n in np.arange(10, round(max_value), 20)]
        plt.yticks(yticks_values, [str(n) + '%' for n in yticks_values], color="grey", size=20)
        plt.ylim(0, max_value)
        # Ind1
        values = df.loc[row].drop(category_label).values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.4)
        # Add a title
        plt.title(title, size=24, color=color, y=1.1, pad=75)
        ax.tick_params(axis='x', pad=45)
        plt.tight_layout()

    # Part 2 : plot radar for each row from df
    # initialize the figure
    plt.figure(figsize=(28, 24))
    # Create a color palette:
    radar_color = plt.cm.get_cmap("tab20b", len(df.index))
    # Plot a radar for each category
    for row in range(len(df.index)):
        plot_radar(row=row, title=df[category_label][row].capitalize(), color=radar_color(row))
    if save_as_img:
        plt.savefig('radars.png')
        
# Subplots


class Subplot:
    
    def __init__(self, shape, fig_size, cols_to_plot):
        self.n_rows, self.n_cols = shape
        self.fig_size = fig_size
        self.cols_to_plot = cols_to_plot
        self.fig = None
        self.axes = None
        self.axes_range = None
        self.build_figure()
           
    def build_figure(self):
        self.fig, self.axes = plt.subplots(self.n_rows, self.n_cols, figsize=self.fig_size)
        self.axes_range = [self.axes[x, y] for x in range(self.n_rows) for y in range(self.n_cols)]
        # Handle redundant axes : compute the number of redundant axes 
        n_redundant_axe_idx = (self.n_rows*self.n_cols) - len(self.cols_to_plot)
        if n_redundant_axe_idx > 0:
            # Compute the number of redundant axes 
            n_redundant_axes = self.axes_range[-n_redundant_axe_idx:]
            # Delete redundant axes
            for redundant_axe in n_redundant_axes:
                redundant_axe.remove()
                
    def plot_data(self, plot_function, df=None, title=None, x_label=None, y_label=None, title_pad=5, x_pad=5, y_pad=5):
        for a, col in zip(self.axes_range, self.cols_to_plot):
            if df is None:
                plot_function(a, col)
            else:
                if plot_function == 'hist':
                    df.hist(col, ax=a)
                # TO DO : add plot types
                else:
                    df.plot(col, ax=a)
            a.set_title(f"{col}" if title is None else title if type(title) is str else title(col), pad=title_pad)
            a.set_xlabel('Value' if x_label is None else x_label, labelpad=x_pad)
            a.set_ylabel('Effectif' if y_label is None else y_label, labelpad=y_pad)
        return
    
    
def plot_hist_subplots(df, cols, shape, fig_size=(14, 18), x_pad=5, y_pad=5, title_pad=15, h=.65, w=.5):
    subplot = Subplot(shape, fig_size, cols) 
    # Plot histogram for each variable
    subplot.plot_data('hist', df, title_pad=5, x_pad=5, y_pad=5)
    # Add space between subplots
    subplot.fig.subplots_adjust(hspace=h, wspace=w)
    return


def plot_pie_subplots(df, col, cats, shape, fig_size=(14, 18), x_pad=5, y_pad=5, title_pad=15, h=.65, w=.5):
    unique_cats = sorted(df[cats].unique().tolist())
    # Build subplot data (fig, axes, delete redundant axes ...)
    subplot = Subplot(shape, fig_size, unique_cats)
    # pie chart data builder (helper)
    data = lambda cat: df[col][df[cats] == cat].value_counts().sort_index()
    # pie chart helper
    pie_func = lambda a, cat: a.pie(data(cat), labels=data(cat).index.map(lambda s: s.upper()), autopct='%1.1f%%')
    # pie chart title helper
    pie_title = lambda cat: f"{cats} {cat} : {col} distribution"
    # Plot pie chart subplots
    subplot.plot_data(pie_func, title=pie_title, title_pad=5, x_pad=5, y_pad=5)
    # Add space between subplots
    subplot.fig.subplots_adjust(hspace=h, wspace=w)
    return


def plot_time_serie_subplots(df, col, cats, shape, alt_col=None, method='count', fig_size=(14, 18), xlabel=None,
                             ylabel='Count', xpad=5,  ypad=5, titlepad=15, h=1.5, w=.2,
                             time_serie_title=None, return_data=False): # h=.65, w=.5
    """
    
    """
    # Unique categories data
    unique_cats = sorted(df[cats].unique().tolist())
    # Time serie data
    alt_col = col if alt_col is None else alt_col
    if method is 'count':
        ts = df.groupby([cats, col])[alt_col].count().reset_index(name=ylabel)
    elif method is 'mean':
        ts = df.groupby([cats, col])[alt_col].mean().reset_index(name=ylabel)
    elif method is 'median':
        ts = df.groupby([cats, col])[alt_col].median().reset_index(name=ylabel)
    elif method is 'max':
        ts = df.groupby([cats, col])[alt_col].max().reset_index(name=ylabel)
    elif method is 'min':
        ts = df.groupby([cats, col])[alt_col].min().reset_index(name=ylabel)
    # Convert date to datetime
    ts[col] = pd.to_datetime(ts[col])
    # Time serie lifetime
    ts['lifetime'] = [(ts[ts[cats] == cat][col].max() - ts[ts[cats] == cat][col].min()).days for cat in ts[cats]]
    # Build subplot data (fig, axes, delete redundant axes ...)
    subplot = Subplot(shape, fig_size, unique_cats)
    # Time serie helper
    plot_func = lambda a, cat: ts[ts[cats] == cat].plot(x=col, y=ylabel, ax=a)
    # Time serie title helper
    get_lifetime = lambda cat: ts[ts[cats] == cat]['lifetime'].unique().item()
    if time_serie_title is None:
        time_serie_title = lambda cat: f"{cats} : {cat} lifetime = {get_lifetime(cat)} days" \
                                     + f"\n\n(≈ {round(get_lifetime(cat)/30)} months)"
    # Time serie subplots
    subplot.plot_data(plot_func,
                      title=time_serie_title,
                      x_label=col if xlabel is None else xlabel,
                      y_label=ylabel,
                      title_pad=titlepad,
                      x_pad=xpad,
                      y_pad=ypad)
    # Add space between subplots
    subplot.fig.subplots_adjust(hspace=h, wspace=w)
    if return_data:
        return ts
    
# Old

# def plot_hist_subplots(df, cols, n_rows, n_cols, fig_size=(14, 18), x_pad=5, y_pad=5, title_pad=15, h=.65, w=.5):
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex=False, sharey=False)
#     axes_range = [axes[x, y] for x in range(n_rows) for y in range(n_cols)]
#     # Plot histogram for each variable
#     for a, col in zip(axes_range, cols):
#         df.hist(col, ax=a)
#         a.set_title(col, pad=title_pad)
#         a.set_xlabel('Value', labelpad=x_pad)
#         a.set_ylabel('Effectif', labelpad=y_pad)
#     # Handle redundant axes
#     # Compute the number of redundant axes 
#     n_redundant_axe_idx = (n_rows*n_cols) - len(cols)
#     if n_redundant_axe_idx > 0:
#         # Compute the number of redundant axes 
#         n_redundant_axes = axes_range[-n_redundant_axe_idx:]
#         # Delete redundant axes
#         for redundant_axe in n_redundant_axes:
#             redundant_axe.remove()
#     # Add space between subplots
#     fig.subplots_adjust(hspace=h, wspace=w)
#     return


# def plot_pie_subplots(df, col, cats, n_rows, n_cols, fig_size=(14, 18), x_pad=5, y_pad=5, title_pad=15, h=.65, w=.5):
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex=False, sharey=False)
#     axes_range = [axes[x, y] for x in range(n_rows) for y in range(n_cols)]
#     # Plot histogram for each variable
#     unique_cats = df[cats].unique().tolist()
#     for a, cat in zip(axes_range, unique_cats):
#         data = df[col][df[cats] == cat].value_counts().sort_index()
#         a.pie(data,
#               labels=[x.upper() if type(x) is str else x.name for x in data.index.tolist()],
#               autopct='%1.1f%%')
#         a.set_title(f"{cats} {cat} : {col} distribution", pad=title_pad)
#         a.set_xlabel('Value', labelpad=x_pad)
#         a.set_ylabel('Effectif', labelpad=y_pad)
#     # Handle redundant axes
#     # Compute the number of redundant axes 
#     n_redundant_axe_idx = (n_rows*n_cols) - len(unique_cats)
#     if n_redundant_axe_idx > 0:
#         # Compute the number of redundant axes 
#         n_redundant_axes = axes_range[-n_redundant_axe_idx:]
#         # Delete redundant axes
#         for redundant_axe in n_redundant_axes:
#             redundant_axe.remove()
#     # Add space between subplots
#     fig.subplots_adjust(hspace=h, wspace=w)
#     return