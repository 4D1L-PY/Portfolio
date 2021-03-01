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

def groupby_plot(df, variable, chart_type='bar', ascending=False, normalize=False):
    group_sorted = df.groupby(variable)[variable].count().sort_values(ascending=ascending)
    if normalize:
        return group_sorted.plot(kind=chart_type, logx=True, logy=True)
    else:
        return group_sorted.plot(kind=chart_type)


def count_plot(df, variable, xlabel='Effectif', ylabel=None, plotsize=(15, 8), sort=False, save_as_img=False):
    if sort:
        df.sort_values(by=[variable], inplace=True)
    plt.figure(figsize=plotsize)
    ax = sns.countplot(y=variable, data=df)
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
            data = [df[cat] for cat in uniques_cat]
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
    for row in range(0, len(df.index)):
        plot_radar(row=row, title=df[category_label][row].capitalize(), color=radar_color(row))
    if save_as_img:
        plt.savefig('radars.png')
