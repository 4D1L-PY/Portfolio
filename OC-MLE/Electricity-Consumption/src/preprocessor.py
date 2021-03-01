from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.feature_selection import SelectFromModel
from src.evaluator import np, pd, plt


########################################################################################################################
#                                            Feature Scaling                                                           #
########################################################################################################################


def standard_scaler(x, center=True, reduce=True, return_std_scaler=False):
    """
    Standardize features by removing the mean and scaling to unit variance (z = (x - u) / s)
    
    :param x: unscaled data (numpy array)
    :param center: center unscaled data (mean = 0)
    :param reduce: reduce unscaled data (standard deviation = 1)
    :param return_std_scaler: boolean value which enable returning (or not) StandardScaler instance
    
    :return: scaled data (numpy array), StandardScaler instance (optional)
    """
    std_scaler = StandardScaler(with_mean=center, with_std=reduce)
    x_scaled = std_scaler.fit_transform(x)
    if return_std_scaler:
        return x_scaled, std_scaler
    return x_scaled


def log_scaler(x, convert_input_data=True, return_log_scaler=False):
    """
    Features log scaling

    :param x: unscaled data (numpy array)
    :param convert_input_data: if True convert input data to a 2-dimensional NumPy array or sparse matrix
    :param return_log_scaler: boolean value which enable returning (or not) FunctionTransformer instance
    
    :return: scaled data (numpy array), FunctionTransformer instance (optional)
    """
    log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=convert_input_data)
    x_scaled = log_transformer.fit_transform(x)
    if return_log_scaler:
        return x_scaled, log_transformer
    return x_scaled


def polynomial_scaler(x, d=2, i=False, biais=True, output_order='C', return_poly_scaler=False):
    """
    Features polynomial scaling

    :param x: unscaled data (numpy array)
    :param d: The degree of the polynomial features scaling (default = 2).
    :param i: If true, only interaction features are produced.
    :param biais: include a bias column (the feature in which all polynomial powers are zero)
    :param output_order: Order of output array in the dense case (default = 'C')
    'F' order is faster to compute, but may slow down subsequent estimators.
    (cf : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
    :param return_poly_scaler: boolean value which enable returning (or not) PolynomialFeatures instance
    
    :return: scaled data (numpy array), PolynomialFeatures instance (optional)
    """
    poly_scaler = PolynomialFeatures(d, i, biais, output_order)
    x_scaled = poly_scaler.fit_transform(x)
    if return_poly_scaler:
        return x_scaled, poly_scaler
    return x_scaled


def reverse_standardization(x_scaled, scaler):
    """
    Inverse standardized features transformation
    
    :param x_scaled: scaled data (numpy array)
    :param scaler: StandardScaler or FunctionTransformer instance
    
    :return: unscaled data
    """
    return scaler.inverse_transform(x_scaled)


########################################################################################################################
#                                              Feature Selection                                                       #
########################################################################################################################


def print_features_reduction(features, features_to_delete, operation_type='features'):
    """
    Display feature reduction ratio
    
    :param features: features count
    :param features_to_delete: features which will be removed from dataset
    :param operation_type: operation type label (filter features with null variance, correlated ...)
    
    """
    total_features = features.shape[1]
    total_features_to_delete = len(features_to_delete)
    reduction_ratio = (total_features_to_delete/total_features)*100
    print('{0}/{1} {2}, reduction of {3:.1f}%'.format(total_features_to_delete,
                                                      total_features,
                                                      operation_type,
                                                      reduction_ratio))
    
    
def features_with_null_variances(df, verbose=False):
    """
    Extract feature labels with null variance
    
    :param df: a dataframe
    :param verbose: display feature reduction ratio
    
    :return: feature labels with null variance (list)
    """
    df_var = df.var()
    cols_null_var = df_var[df_var == 0].index.tolist()
    if verbose:
        print_features_reduction(df, cols_null_var, 'features with null variance')
    return cols_null_var


def features_with_identical_variances(df, col_kept='last', verbose=False):
    """
    Extract feature labels with identical variance (redundant features)
    
    :param df: a dataframe
    :param col_kept: redundant feature selected ('last' mean last occurrence)
    (cf : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html)
    :param verbose: display feature reduction ratio
    
    :return: feature labels with null variance (list)
    """
    # Build features variances dictionary
    df_var = dict(df.var())
    # Build features unique variances dictionary 
    df_unique_var = dict(df.var().drop_duplicates(keep=col_kept))
    # Get the difference between dictionaries 
    redundant_cols = list(dict(df_var.items() ^ df_unique_var.items()).keys())
    if verbose:
        print_features_reduction(df, redundant_cols, 'features with identical variance')
    return redundant_cols


def filter_invalid_variances(df, feature_kept='last', v=True):
    """
    Filter features which have invalid variances
    
    :param df: a dataframe
    :param feature_kept: redundant feature selected ('last' mean last occurrence)
    :param v: display feature reduction ratio
    
    :return: a dataframe with filtered features
    """
    null_variance_cols = features_with_null_variances(df, v)
    identical_variance_cols = features_with_identical_variances(df, feature_kept, v)
    invalid_variances_cols = null_variance_cols + identical_variance_cols
    return df[[col for col in df.columns if col not in invalid_variances_cols]]


def filter_correlated_features(df, threshold=0.5, verbose=False, return_corr_features=False):
    """
    Filter correlated features
    
    :param df: a dataframe
    :param threshold: Pearson correlation coefficient value
    :param verbose: display feature reduction ratio
    :param return_corr_features: boolean value which enable/disable returning correlated features list
    
    :return: a dataframe with filtered features, correlated features list (optional)
    """
    # Build correlation matrix
    corr_matrix = df.corr()
    # Get the upper triangle of correlation matrix
    boolean_matrix = np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
    upper_corr_matrix = corr_matrix.where(boolean_matrix)
    # Find index of feature columns with correlation greater than 0.5
    corr_features = [col for col in upper_corr_matrix.columns if any(abs(upper_corr_matrix[col]) > threshold)]
    if verbose:
        print_features_reduction(df, corr_features, 'features correlated')
    # Filter features
    df = df[[col for col in df.columns if col not in corr_features]]
    if return_corr_features:
        return df, corr_features
    return df


def filter_features_by_threshold(features, x_train, x_test, model, threshold='median', model_prefit=True, verbose=False):
    """
    Filter training and testing feature sets based on a specific threshold
    doc : https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html

    :param features: a dataframe which contains features data
    :param x_train: training set
    :param x_test: testing set
    :param model: sklearn model object
    :param threshold: threshold value (float)
    :param model_prefit: Whether a prefit model is expected to be passed into the constructor directly or not (boolean)
    :param verbose: display feature reduction ratio
    
    :return: filtered training and testing feature sets, selected features labels & index lists (dict)
    """
    selector = SelectFromModel(model, prefit=model_prefit, threshold=threshold)
    # Get selected features index
    selected_features_idx = selector.get_support()
    # Filter training & testing data
    x_train_filtered = selector.transform(x_train)
    x_test_filtered = selector.transform(x_test)
    selected_features_labels = features.iloc[:, selected_features_idx].columns.tolist()
    selected_features_dict = {'X_train': x_train_filtered,
                              'X_test': x_test_filtered,
                              'labels': selected_features_labels,
                              'idx': selected_features_idx}
    if verbose:
        print_features_reduction(features, selected_features_labels, 'selected features')
    return selected_features_dict


def get_features_importance(labels, coefs, abs_coefs=False, non_zero_coefs=False, sort=True, verbose=True):
    """
    Build features importance dataframe based on features labels & values/coefficients

    :param labels:
    :param coefs:
    :param abs_coefs:
    :param non_zero_coefs:
    :param sort:
    :param verbose: display selected features number

    :return: a dataframe with features importance
    """
    # Build feature importance dataframe
    fimp_df = pd.DataFrame({'feature': labels, 'coefficient': coefs})
    # Get positive coefficients
    if abs_coefs:  
        fimp_df['coefficient'] = np.abs(fimp_df['coefficient'])
    # Filter zero coefficients
    if non_zero_coefs:
        fimp_df = fimp_df[fimp_df['coefficient'] != 0]
    # Sort features (get most important features at the head)
    if sort:
        fimp_df = fimp_df.sort_values('coefficient', ascending=False).reset_index(drop=True)
    # Print selected features and reduction ratio
    if verbose:
        # Get features count
        total_features = len(labels)
        # Get filtered features count (zero coefficients removed)
        selected_features = fimp_df.shape[0]
        reduction_ratio = (1 - (selected_features/total_features))*100
        print('{0}/{1} features selected, reduction of {2:.1f}%'.format(selected_features,
                                                                        total_features,
                                                                        reduction_ratio))
    # Compute coefficient frequency in order to calculate cumulative feature importance
    coefficients_sum = fimp_df['coefficient'].sum()
    fimp_df['coefficient_frequency'] = fimp_df['coefficient'] / coefficients_sum
    fimp_df['cumulative_coefficient_frequency'] = np.cumsum(fimp_df['coefficient_frequency'])
    return fimp_df


def plot_n_top_features(features, model_label, n=10, x_label='feature', y_label='coefficient', plot_size=(12, 4)):
    """

    :param features: a dataframe which contains features data
    :param model_label:
    :param n:
    :param x_label:
    :param y_label:
    :param plot_size:
    :return:
    """
    features.head(n).plot(x=x_label, y=y_label, kind='barh', figsize=plot_size)
    plt.gca().invert_yaxis()
    plt.title(f'{model_label} Top {n} Features', pad=20)
    plt.xlabel('Coefficients', labelpad=20)
    plt.ylabel('Features labels', labelpad=20)
    plt.show()
    
    
def plot_cumulative_features_importance(features, threshold=0.90, plot_size=(12, 8)):
    """

    :param features: a dataframe which contains features data
    :param threshold:
    :param plot_size:
    :return:
    """
    plt.figure(figsize=plot_size)
    # Number of features needed for threshold cumulative importance
    importance_idx = np.min(np.where(features['cumulative_coefficient_frequency'] > threshold))
    thr_percentage = 100 * threshold
    l = '{} features required for {:.0f}% of cumulative importance.'.format(importance_idx+1, thr_percentage)
    # Cumulative importance plot
    plt.plot(range(len(features)), features['cumulative_coefficient_frequency'], 'b-', label=l)
    plt.xlabel('Number of Features', fontsize=12, labelpad=20)
    plt.ylabel('Cumulative Coefficient frequency', fontsize=12, labelpad=20) 
    plt.title('Cumulative Feature Importance', fontsize=14, pad=20)
    # plt.title(f'Cumulative Feature Importance\n\n{l}', fontsize=14, pad=20)
    # Threshold  vertical line plot
    plt.vlines(importance_idx + 1, ymin=0, ymax=1.05, linestyles='--', colors='red')
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.show()
