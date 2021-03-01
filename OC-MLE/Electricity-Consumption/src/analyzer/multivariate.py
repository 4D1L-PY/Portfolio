from src.analyzer.univariate import np, plt, sns
import scipy.stats as st
from statsmodels.formula.api import ols


########################################################################################################################
#                                            Multivariate Analysis                                                     #
########################################################################################################################

# 2 quantitatives variables


def correlation_matrix(df, as_chart=True, precision=2, title=None, rotate=90, save_as_img=False, size=(16, 12)):
    """
    """
    corr = df.corr()
    if as_chart:
        colormap = plt.cm.RdBu
        plt.figure(figsize=size)
        if title is None:
            title = 'Pearson Correlation of Features'
        plt.title(title, y=1.05, size=15, pad=20)
        mask = np.triu(np.ones_like(corr, dtype=np.bool))
        ax = sns.heatmap(corr, linewidths=0.5, vmax=1.0, square=True, cmap=colormap,
                         linecolor='white', annot=True, mask=mask, cbar_kws={"shrink": .5},
                         fmt='.{}f'.format(precision))

        ax.set_xlim(0, df.shape[1] - 1)
        ax.set_ylim(df.shape[1], 1)
        plt.xticks(rotation=rotate)
        if save_as_img:
            plt.tight_layout()
            plt.savefig('corr_matrix.jpg')
        plt.show()
    else:
        return corr.style.background_gradient(cmap='coolwarm').set_precision(precision)


def get_top_n_correlations(df, abs_corr=False, n=5):
    """
    cf : https://code.i-harness.com/en/q/10f46da
    """
    # Get diagonal and lower triangular pairs of correlation matrix
    labels_to_drop = set()
    cols = df.columns
    for i in range(df.shape[1]):
        for j in range(0, i + 1):
            labels_to_drop.add((cols[i], cols[j]))
    # Get the first n correlations
    au_corr = df.corr().abs().unstack() if abs_corr is True else df.corr().unstack()
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[:n]


def pearson_correlation(df, x, y, precision=2):
    r = round(st.pearsonr(df[x], df[y])[0], precision)
    covariance = round(np.cov(df[x], df[y])[1, 0], precision)
    return r, covariance


def plot_linear_regression(df, x, y, save_as_img=False):
    ax = sns.regplot(x, y, data=df)
    ax.set_xlabel(x, labelpad=20)
    ax.set_ylabel(y, labelpad=20)
    ax.set_title('Linear regression ({} & {})'.format(x, y), pad=20)
    if save_as_img:
        plt.tight_layout()
        plt.savefig('{}_{}_linear_reg.jpg'.format(x, y))


def plot_residuals(df, x, y, save_as_img=False):
    ax = sns.residplot(x, y, lowess=True, data=df)
    ax.set_xlabel(x, labelpad=20)
    ax.set_ylabel(y, labelpad=20)
    ax.set_title('Residuals dispersion', pad=20)
    if save_as_img:
        plt.tight_layout()
        plt.savefig('{}_{}_residplot.jpg'.format(x, y))


# 1 qualitative + 1 quantitative variables

def anova_correlation_ratio(df, X, Y, precision=2):
    """
    :param: X: qualitative variable
    :param: Y: quantitative variable
    """
    y_mean = df[Y].mean()
    classes = []
    for cls in df[X].unique():
        yi_cls = df[Y][df[X] == cls]
        classes.append({'ni': len(yi_cls), 'class_mean': yi_cls.mean()})
    # Total Sum of Squares
    TSS = sum([(yj - y_mean) ** 2 for yj in df[Y]])  # SCT
    # Sum of Squares of the Model
    SSM = sum([c['ni'] * (c['class_mean'] - y_mean) ** 2 for c in classes])  # SCE
    return round(SSM / TSS, precision)


def anova_stats_model(df, x, y):
    model_fit = ols('{} ~ C({})'.format(x, y), data=df).fit()
    r2, pvalue = round(model_fit.rsquared, 2), round(model_fit.f_pvalue, 2)
    return r2, pvalue, model_fit


def anova_test(df, numeric_col, cat_col, alpha=0.05, verbose=False):
    df_anova = df[[cat_col, numeric_col]]
    categories = df[cat_col].unique().tolist()
    anova_cat_data = {cat: df_anova[numeric_col][df_anova[cat_col] == cat] for cat in categories}
    stat, p = st.f_oneway(*anova_cat_data.values())
    p = round(p, 3)
    if p < alpha:
        return f'Reject H0 (p-value < {alpha})'
    else :
        return 'Accept H0'


def residue_normality_test(model, test_type='shapiro'):
    if test_type is 'shapiro':
        w, p_value = st.shapiro(model.resid)
        return p_value
    elif test_type is 'npp':  # normal probability plot ("droite de Henry")
        st.probplot(model.resid, plot=plt)
    else:
        raise Exception('{} not valid'.format(test_type))


def homoscedasticity_test(df, X, Y, test_type='levene'):
    groups = [df[df[X] == cls][Y] for cls in df[X].unique()]
    if test_type is 'levene':
        levene, p_value = st.levene(*groups)
    elif test_type is 'bartlett':
        bartlett, p_value = st.bartlett(*groups)
    else:
        raise Exception('{} not valid'.format(test_type))
    return p_value


# 2 qualitatives variables


def contingency_table(df, X, Y):
    return df[[X, Y]].pivot_table(index=X, columns=Y, aggfunc=len, margins=True, margins_name="Total")


def chi_2_heatmap(df, X, Y, save_as_img=False, title='Chi-2 contingency table'):
    cont = contingency_table(df, X, Y)
    # Chi-2
    ni = cont.loc[:, ["Total"]]  # ni
    nj = cont.loc[["Total"], :]
    n = len(df)
    indep = ni.dot(nj) / n
    nij = cont.fillna(0)
    xi_ij = (nij - indep) ** 2 / indep
    xi_n = xi_ij.sum().sum()
    table = xi_ij / xi_n
    ax = sns.heatmap(table.iloc[:-1, :-1], annot=nij.iloc[:-1, :-1], cmap='Blues', fmt='g')
    ax.set_xlabel(Y, labelpad=20)
    ax.set_ylabel(X, labelpad=20)
    ax.set_title(title, pad=20)
    if save_as_img:
        plt.tight_layout()
        plt.savefig('{}_{}_chi2_heatmap.jpg'.format(X, Y))

        
def chi_2_test(df, X, Y, title='Chi-2 contingency table', alpha=0.05, verbose=False, zoom=False, zoom_size=(12,8)):
    chi_2_subset = df.copy()
    chi_2_heatmap(chi_2_subset, X, Y, False, title)
    if zoom:
        sns.set(rc={'figure.figsize': zoom_size})
    cont = contingency_table(chi_2_subset, X, Y).fillna(0)
    chi2, p, dof, expected_freq = st.chi2_contingency(cont)
    if verbose:
        print("p value = {}".format(p))
    if p < alpha:
        return 'Reject H0'
    else :
        return 'Accept H0'