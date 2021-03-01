import sys
import geocoder
sys.path.append("..")
from glob import glob
from itertools import combinations
from operator import attrgetter
from geopy import distance
from tabulate import tabulate
# Personal package named src
from src.datacleaner import *
# Data preprocessing
from src.preprocessor import *
# Model evaluation
from src.evaluator import *
# EDA
from src.analyzer.univariate import *
from src.analyzer.multivariate import *
# Dimensionality reduction
from src.dimensionality_reducer.pca import LinearPCA
from src.dimensionality_reducer.tsne import tSNE
# Clustering
from src.clusterizer.cah import CAH
from src.clusterizer.kmeans import Kmeans
# Model selection
from sklearn.model_selection import train_test_split
# Classifiers
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


########################################################################################################################
#                                  EDA IPYTHON NOTEBOOK HELPERS                                                        #
########################################################################################################################


# Main data-cleaning helpers


def csv_reader(keyword, folder_path='data/csv/', verbose=True):
    """
    Import csv file based on a keyword that refers to its file type
    
    :param: keyword: a keyword which qualify a csv file
    :param: folder_path: relative path to csv files 
    :param: verbose: boolean which enable/disable verbose mode
    
    :return: tuple which contains a dataframe and csv filename
    
    """
    csv_files = glob(f'{folder_path}*.csv')
    csv_file_selected = [file for file in csv_files if keyword in file][0]
    csv_file_name = csv_file_selected.replace(folder_path, '')
    df = pd.read_csv(csv_file_selected)
    if verbose:
        print(f'CSV file selected : {csv_file_name}')
    return df, csv_file_name


def iterative_merge(df, dfs, keys, operations, left_suffix, verbose=True):
    """
    Merge multiple dataframes iteratively
    
    :param: df: a dataframe
    :param: dfs: a list of dataframes
    :param: keys: a list of primary or foreign keys
    :param: left_suffix: suffix to use from left frame’s overlapping columns
    
    :return: a completly merged dataframe
    
    """
    df_merged = ''
    for df_from_list, k, op in zip(dfs, keys, operations):
        if df_from_list is dfs[-1]:
            df_merged = df.join(df_from_list, on=k, how=op, lsuffix=left_suffix)
        else:
            df_merged = df.merge(df_from_list, on=k, how=op)
        if verbose:
            get_data_loss(df_merged, df_from_list if df_from_list is dfs[0] else df)
        df = df_merged
    return df_merged


def decompose_datetime(df, cols, convert_to_datetime=True, autodelete=False):
    """
    Decompose datetime feature by multiple time scales (year, month, day, hour, timestamp)
    (increase feature number by 5n)
    
    :param: df: a dataframe
    :param: cols: a list of features
    :param: convert_to_datetime: boolean value which enable/disable feature conversion to datetime
    :param: autodelete: boolean value which enable/disable deleting original features
    
    """
    # Multiply feature number by 5
    for col in cols:
        rebuilt_col = "_".join(col.split('_')[:-1])
        if convert_to_datetime:
            df[col] = pd.to_datetime(df[col])
        if col is not 'order_approved_at':
            df[rebuilt_col+'_year'] = df[col].dt.year
            df[rebuilt_col+'_month'] = df[col].dt.month
            df[rebuilt_col+'_day'] = df[col].dt.day
            df[rebuilt_col+'_hour'] = df[col].dt.hour
        df[rebuilt_col+'_ts'] = df[col].values.astype(np.int64) // 10 ** 9
    if autodelete:
        delete_cols(df, cols)
        
        
def decompose_variable_by_time_scale(df, time_scale):
    """
    Decompose datetime variable by time scale (year, quarter, month ...)
    as dummy variables
    
    :param: df: a dataframe
    :param: time_scale: a time scale label ('year', 'quarter', 'month')
    
    :return: a dataframe with dummy datetime variables
    
    """
    # (Optional) Build exception
    time_scale_exception = Exception(f'TimeScaleError : {time_scale} is an invalid value')
    # Excluded string patterns (shipping & estimated delivery variables)
    patterns = ['limit', 'estimated']
    # Time scale suffix for the first condition (scope variable by time scale)
    suffix = '_ts' if time_scale is 'quarter' else time_scale
    # Filter variable based on suffix and excluded string patterns conditions
    filtered_cols = [c for c in df.columns if (c.endswith(suffix) & all(w not in c for w in patterns))]
    # Create indicator/dummy variables for each filtered variables
    for col in filtered_cols:
        # Create a temporary variable label (e.g : 'order_purchase')
        temp_var_label = "_".join(col.split('_')[:-1])
        # Build a Serie which contains transformed timestamps as quarters (e.g : '2017Q1')
        if time_scale is 'quarter':
            df[temp_var_label] = pd.to_datetime(df[col], unit='s').dt.to_period(freq='Q').map(str)
        # Build a Serie which contains transformed int as month (e.g : 'January')
        elif time_scale is 'month':
             df[temp_var_label] = pd.to_datetime(df[col], format='%m').dt.month_name().map(lambda m: m.lower())
        # Build a Serie which contains transformed int as year (e.g : 2017)
        elif time_scale is 'year':
            df[temp_var_label] = df[col]
        else:
            raise(time_scale_exception)
        # Extract uniques values from dataframe
        time_scale_cols = [f'{temp_var_label}_{s}' for s in df[temp_var_label].unique()]
        # Generate indicator variables (e.g : 'order_purchase_2017Q1')
        # N.B : temporary variable is not deleted in order to aggregate the data with it later
        df = get_dummy_indicators(df, temp_var_label, drop=False, prefix_origin_col=True)
        # Remove duplicated columns
        df = df.loc[:, ~df.columns.duplicated()]
        # Create indicator variable for each time scale unique values
        for time_scale_col in time_scale_cols:
            # (e.g : 'order_purchase_2016Q4')
            var_time_scale_order_label = f'{time_scale_col}'
            if time_scale is 'month':
                # Aggregate order values by time scale (year + month) and customer id
                year_col = col.replace('month', 'year')
                # Define groupby column labels
                gp_cols = [year_col, time_scale_col, 'customer_id']
            elif time_scale in ['quarter', 'year']:
                gp_cols = [time_scale_col, 'customer_id']
            else:
                raise(time_scale_exception)
            # Build variable which concerns the total amount of payments by time scale
            if temp_var_label == 'order_purchase':
                var_time_scale_payment_label = temp_var_label.replace('order_purchase', 'total_customer_payment')
                # Aggregate payment values by time scale and customer id (and convert Series to dictionary)
                time_scale_payment_gp_dict = dict(df.groupby(gp_cols)['payment_value'].sum())
                df[var_time_scale_payment_label] = df.set_index(gp_cols).index.map(time_scale_payment_gp_dict.get)
            # Build variable which concerns the total number of orders by time scale
            # Aggregate order values by time scale and customer id (and convert Series to dictionary)
            time_scale_order_gp_dict = dict(df.groupby(gp_cols)['order_id'].count())
            # Create variable by mapping customer time scale groupby dictionary
            df[var_time_scale_order_label] = df.set_index(gp_cols).index.map(time_scale_order_gp_dict.get)
        # Drop temporary variable
        delete_cols(df, temp_var_label)
    return df


def get_geodesic_dist(origin_coords, dest_coords, unit='m'):
    """
    Compute geodesic distance between 2 locations
    
    :param origin_coords: origin coordinates (Series)
    :param dest_coords: destination coordinates (Series)
    :param unit: distance unit value (km or m)
    
    source :
     - https://github.com/geopy/geopy
     
    :return: a distance (in kilometers or meters)
    
    """
    dist = distance.geodesic(dest_coords, origin_coords)
    return dist.km if unit is 'km' else dist.m if unit is 'm' else None


def eval_delivery_date(x):
    """
    Categorize delivery evaluation based on a range of values between [-1, 1]
    
    :param x: delivery evaluation value
    
    :return: a delivery evaluation category
    
    """
    if x == 0:
        return 'matched'
    elif x > 0:
        return 'overvalued'
    else:
        return 'undervalued'
    
    
def plot_time_serie(df, col, time_unit='s', agg_label='orders', s=(12,8),
                    rot=0, xlab='Date', time_filter=None, return_data=False):
    """
    Plot time serie
    
    :param: df: a dataframe
    :param: col: the column from which we make the time series
    :param: time_unit: time unit (second, minute, hour ...)
    :param: agg_label: the column from which the aggregation is carried out
    :param: s: figure size
    :param: rot: rotation value applied to x axis
    :param: xlab: x axis label
    :param: time_filter: a time filter (year, month, day ...)
    :param: return_data: boolean which enable/disable returning time serie data
    
    :return: time serie data (optional)
    
    """
    order_date_col = 'order_purchase_date'
    df[order_date_col] = pd.to_datetime(df[col], unit=time_unit).dt.date
    gp = df.groupby(order_date_col)[order_date_col].count().reset_index(name=agg_label)
    if time_filter is not None:
        gp = gp[gp[order_date_col].map(str).str.contains(str(time_filter))]
    gp.plot(x='order_purchase_date', y=agg_label, figsize=s)
    plt.xticks(rotation=rot)
    plt.xlabel(xlab, labelpad=20)
    plt.tight_layout()
    plt.show()
    if return_data:
        return gp
    return


def plot_hist_by_cat(df, num_var, cat_var, s=(8, 6), legend_size=8, legend_loc='upper left'):
    """
    Plot histogram by categories
    
    :param: df: a dataframe
    :param: num_var: y axis variable
    :param: cat_var: x axis variable
    :param: s: figure size
    :param: legend_size: legend label size
    :param: legend_loc: legend location
    
    """
    plt.figure(figsize=s)
    for cat in df[cat_var].unique():
        sns.distplot(df[df[cat_var]==cat][num_var], label=cat)
    plt.legend(loc=legend_loc, fontsize=legend_size)
    if '_' in num_var and '_' in cat_var:
        num_var_rebuilt, cat_var_rebuilt = [" ".join(v.split('_')) for v in [num_var, cat_var]]
        title = f'{num_var_rebuilt.capitalize()} distribution by {cat_var_rebuilt}'
    else:
        title = f'{num_var.capitalize()} distribution by {cat_var}'
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return


# Customer segmentation


def RFM_score(x, p, d):
    """
    Compute RFM score based on quantile values
    
    :param x: value from RFM dataframe
    :param p: RFM class (recency, frequency, monetary_value)
    :param d: quantile dictionary
    
    :return: RFM score (from 1 to n+1 where n is the total quantile classes)
    
    """
    q_range = list(list(d.values())[0].keys())
    rfm_score = 1
    rfm_score_reversed = len(q_range)+1
    for q in q_range:
        if p is 'recency':
            if x <= d[p][q]:
                return rfm_score_reversed
        elif p in ['frequency', 'monetary_value']:
            if x <= d[p][q]:
                return rfm_score
        rfm_score += 1
        rfm_score_reversed -= 1
    return rfm_score_reversed if p is 'recency' else rfm_score


def plot_RFM_segments(df, segments, n=None, n_dim=2, p=30, tsne_init='random', v=True, plt_size=(8, 6)):
    """
    Plot RFM segments with t-SNE
    
    :param: df: a dataframe
    :param: segments: segment values (Series)
    :param: n: select the n customers by segment (optional)
    :param: n_dim: number of dimensions to display
    :param: p: perplexity value
    :param: tsne_init: t-SNE initialization ('pca' or 'random')
    :param: v: boolean which enable/disable verbose mode
    :param: plt_size: figure size
    
    """
    df_tsne = df.copy()
    df_tsne.loc[:, 'segment'] = segments
    if n is not None:
        df_tsne.reset_index(inplace=True)
        # Group by segment
        sort_by_recency_helper = lambda x: x.sort_values(["recency"], ascending=False)
        grouped_segments = df_tsne.groupby(["segment"]).apply(sort_by_recency_helper).reset_index(drop=True)
        # Reduce data in order to plot segments with t-SNE
        reduced_segments = grouped_segments.groupby(["segment"])["segment", "customer_id"].head(n)
        df_tsne = df_tsne[df_tsne.loc[:, 'customer_id'].isin(reduced_segments.loc[:, 'customer_id'])]
        df_tsne.set_index('customer_id', inplace=True)
        segments = df_tsne.segment.values    
    X = df_tsne.iloc[:, :-1]
    # Standardize data
    df_scaled, std_scaler = standard_scaler(X, return_std_scaler=True, rebuild_df=True)
    tsne = tSNE(df_scaled, n_comp=n_dim, perplexity=p, init=tsne_init, verbose=v)
    tsne.fit()
    tsne.plot(plot_size=plt_size, cluster_labels=segments)
    return 


# Cohort analysis


def build_retention_matrix(df, cohort_cols):
    """
    Retention matrix helper
    
    :param: df: a dataframe
    :param: cohort_cols: list of variables needed to build retention matrix
    
    :return: retention matrix and cohort size
    
    """
    # Decompose cohort features
    user_col, order_col, date_col = cohort_cols
    # Build cohort subset & remove duplicates
    df_cohort_subset = df.loc[:, cohort_cols].drop_duplicates()
    # Transform order purchase date to month period
    df_cohort_subset['order_month'] = df_cohort_subset[date_col].dt.to_period('M')
    # Aggregate cohort subset by user id
    df_cohort_subset['Cohort'] = df_cohort_subset.groupby(user_col)[date_col].transform('min').dt.to_period('M') 
    # Aggregate cohort subset by cohort, order month & unique user id
    df_cohort_gp = df_cohort_subset.groupby(['Cohort', 'order_month']).agg(n_customers=(user_col,
                                                                                        'nunique'))
    # Reset cohort subset index & build cohort dataframe
    df_cohort = df_cohort_gp.reset_index()
    # Create period number feature from difference between order month and cohort month
    df_cohort['period_number'] = (df_cohort.order_month - df_cohort.Cohort).apply(attrgetter('n')) + 1
    # Build cohort pivot table
    cohort_pivot = df_cohort.pivot_table(index='Cohort', columns='period_number', values='n_customers')
    # Get cohort size
    cohort_size = cohort_pivot.iloc[:,0]
    # Build retention matrix 
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0) # * 100
    # Return retention matrix (DataFrame) & cohort size (Series) 
    return retention_matrix, cohort_size


def plot_cohort_analysis(cohort_data, cohort_size, threshold=None):
    """
    Plot retention matrix based on cohort data & size
    
    :param cohort_data: a dataframe which represent retention matrix
    :param cohort_size: a serie which store cohort size 
    :param threshold: threshold ratio which filter retention values
    
    """
    sns.axes_style("white")
    # Generate two subplots : cohort retention & cohort size heatmaps
    fig, ax = plt.subplots(1,
                           2,
                           figsize=(12, 8),
                           sharey=True,
                           gridspec_kw={'width_ratios': [1, 11]})
    # Cohort analysis heatmap
    sns.heatmap(cohort_data,
                mask=cohort_data.isnull(),
                annot=True,
                fmt='.0%',
                cmap='Blues',
                ax=ax[1])
    # Filter cohort data by threshold value
    sns.heatmap(cohort_data,
                mask=cohort_data.iloc[:, 1:] >= threshold,
                cmap='plasma_r',
                cbar=False,
                ax=ax[1])
    # Add title & axis labels
    ax[1].set_title('Customer Retention Rate (Monthly)',
                    fontsize=14,
                    pad=25)
    ax[1].set_xlabel('Cohort Period', labelpad=15)
    ax[1].set_ylabel('Retention', labelpad=15)
    # Cohort size data
    cohort_size_df = pd.DataFrame(cohort_size).rename(columns={1: 'Cohort Size'})
    # Cohort size heatmap
    sns.heatmap(cohort_size_df,
                annot=True,
                cbar=False,
                fmt='g',
                cmap='BuGn',
                ax=ax[0])
    # Improve plot layout
    fig.tight_layout()
    return


# Preprocessing modelization dataset


def frequency_count(df, col='customer_id', precision=3):
    """
    Build frequency distribution
    
    :param: df: a dataframe
    :param col: variable to aggregate as unique count
    :param precision: rounding precision value
    
    :return: a transposed dataframe
    """
    grouped_frequency = df.groupby('frequency')[col].nunique().to_frame(name='count')
    grouped_frequency_percentage = (grouped_frequency['count'] / grouped_frequency['count'].sum())*100
    grouped_frequency['%'] = np.round(grouped_frequency_percentage, precision)
    return grouped_frequency.T


def normalize_frequency(df, sampling_frac, verbose=True):
    """
    Normalize frequency distribution by sampling over-represented modality
    
    :param: df: a dataframe
    :param: sampling_frac: float value which represent sampling fraction
    :param verbose: boolean which enable/disable verbose mode
    
    :return: a dataframe with normalize frequency
    
    """
    df_once_orders = df[df['frequency'] == 1]
    df_twice_orders = df[df['frequency'] == 2]
    df_multi_orders = df[df['frequency'] > 2]
    # Sampling
    freq_data = [df_once_orders, df_twice_orders, df_multi_orders]
    sampled_freq_list = [f.sample(frac=s) for f, s in zip(freq_data, sampling_frac)]
    if verbose:
        print('Once orders sample size : {}'.format(sampled_freq_list[0].shape[0]))
        print('Twice orders sample size : {}'.format(sampled_freq_list[1].shape[0]))
    df_recomposed = pd.concat(sampled_freq_list)
    return df_recomposed


def log_scaling(df, cols, pickle_scaler=True, pickle_path='data/pkl'):
    """
    Log scaling helper
    
    :param: df: a dataframe
    :param: cols: a list of features
    :param: pickle_scaler: boolean which enable/disable pickling log scaler instance
    :param: pickle_path: pickling relative path
    
    :return: a log scaled dataframe
    
    """
    log_transformer = None
    for col in cols:
        reshaped_col_data = df.loc[:, col].values.reshape(-1, 1)
        if log_transformer is None:
            df.loc[:, col], log_transformer = log_scaler(reshaped_col_data,
                                                         return_log_scaler=True)
        else:
            df.loc[:, col] = log_transformer.transform(reshaped_col_data)
    # log_scaling_data = {'scaler': log_transformer, 'scaled_cols': cols}
    if pickle_scaler:
        pickle_data(filename='log_transformer',
                    data=log_transformer,
                    folder=pickle_path,
                    method='w')
    return df


########################################################################################################################
#                                 MODELIZATION IPYTHON NOTEBOOK HELPERS                                                #
########################################################################################################################


# TO DO : add scorer param
def get_optimal_kmeans_from_k_range(df, cluster_range=(2, 16), n_best_scores=2):
    """
    Find optimal k-means model from specific k range, evaluated with Silhouette score
    
    :param df: a dataframe
    :param cluster_range: a range of k clusters (data structures allowed : tuple, list, int)
    :param n_best_scores: the n best scores selected (int)
    
    :return: a results dictionary which contains training results dataframes & best model instance
    
    """
    # Train k-means for each cluster number in range
    n_clusters = [cluster_range] if type(cluster_range) is int else range(*cluster_range)
    results = {'n': n_clusters,
               'score': [],
               #'ARI': [],
               'model': [],
               'features': []}
    for n in n_clusters:
        km = Kmeans(df, k_range=n)
        km.fit()
        score = round(silhouette_score(km.X, km.clusters), 2)
        results['model'].append(km)
        results['features'].append(list(km.features))
        results['score'].append(score)
    # Build k-means results
    df_results = pd.DataFrame(results)
    # Get n best scores (largest Silhouette score)
    df_best_results = df_results.nlargest(n_best_scores, 'score')
    # Get total features count
    df_best_results['total_features'] = df_best_results['features'].map(len)
    # Get best model instance
    best_model_instance = df_best_results['model'].iloc[0]
    # Build training results dictionary
    training_results_data = {'all_results': df_results,
                             'best_results': df_best_results,
                             'best_model': best_model_instance}
    return training_results_data


def get_optimal_kmeans_feature_set(df, k_range=(2, 16), verbose=False):
    """
    Find k-means best feature selection evaluated with Silhouette score 
    
    :param df: a dataframe
    :param k_range: a range of k clusters (data structures allowed : tuple, list, int)
    :param verbose: boolean which enable/disable verbose mode
    
    :return: training results from training_cycle function
    
    """
    df_tmp = df.copy()
    # Inner function which start a training cycle based on feature selection

    def training_cycle(df_tmp, feature_start=3, best_score=0, results=[]):
        """
        Train k-means iteratively according to a selection of features updated recursively
        
        :param df_tmp: a dataframe
        :param feature_start: feature index number
        :param best_score: best model score (highest Silhouette score value)
        :param results: list of training cycle results
        
        :return: training results (dataframe)
        
        """
        # Feature selection (list)
        features = df_tmp.columns.tolist()
        # Size of the set of features
        total_features = len(features)
        # k-means training cycle (iterating over the size of the set of features)
        for i in range(feature_start, total_features+1):
            # Selection of a feature slice 
            df_tmp_slice = df_tmp.iloc[:, :i:]
            # Train k-means based on a cluster range (select the first best score for each cluster number)
            km_results = get_optimal_kmeans_from_k_range(df_tmp_slice,
                                                         cluster_range=k_range,
                                                         n_best_scores=1)
            # Get k-means best results 
            km_results_df = km_results['best_results']
            # Get k-means best score
            km_score = km_results_df['score'].iloc[0]
            if verbose:
                km_main_results = km_results_df.loc[:, km_results_df.columns != 'model']
                print(tabulate([list(row) for row in km_main_results.values],
                               headers=list(km_main_results.columns)))
                print('\n')
            # Update best model score
            if best_score <= km_score:
                best_score = km_score
                # Add k-means best results dataframe to training results
                results.append(km_results_df)
            # Drop irrelevant feature from training data
            else:
                last_feature = df_tmp_slice.columns.tolist()[-1]
                delete_cols(df_tmp, last_feature)
                # Restart training cycle based on reduced set of features
                return training_cycle(df_tmp, i, best_score, results)
        # return training results (dataframe)
        return pd.concat(results)          
    return training_cycle(df_tmp)


def highlight_segments(df):
    """
    highlight segments according to specific modalities (order frequency, period, season ...)
    
    :param df: a dataframe
    
    :return: an enhanced dataframe (colored)
    
    """
    # Copy original data
    df_colored = df.copy()
    # Numerical variables conditions
    num_values_masks = {'background-color: mediumblue': df_colored.values == "very high",
                        'background-color: dodgerblue': df_colored.values == "high",
                        'background-color: orange': df_colored.values == "medium",
                        'background-color: salmon': df_colored.values == "low",
                        'background-color: crimson': df_colored.values == "very low"}
    # Season conditions
    season_masks = {'background-color: springgreen': df_colored.values == "spring",
                    'background-color: peru': df_colored.values == "autumn",
                    'background-color: aqua': df_colored.values == "winter",
                    'background-color: yellow': df_colored.values == "summer"}
    # Month period
    period_masks = {'background-color: lightsteelblue': df_colored.values == "beginning",
                    'background-color: steelblue': df_colored.values == "mid-month"}
    # List of all conditions
    masks = [num_values_masks, season_masks, period_masks]
    # Apply colors based on conditions
    for mask in masks:
        for mask_color, mask_cond in mask.items():
            df_colored[mask_cond] = mask_color
    return df_colored 


def highlight_segment_update(df, col='Lifetime (months)'):
    """
    highlight segments according to its lifetime
    
    :param df: a dataframe
    :param col: column label to highlight
    
    :return: an enhanced dataframe (colored)
    
    """    
    # copy df to new - original data are not changed
    df_colored = df.copy()
    masks = {'background-color: tomato': df_colored[col].values <= 10,
             'background-color: darkorange': ((df_colored[col].values > 10)
                                              & (df_colored[col].values < 16)),
             'background-color: limegreen': df_colored[col].values >= 16}
    for mask_color, mask_cond in masks.items():
        df_colored[mask_cond] = mask_color
    return df_colored


def feature_importance_ratio(df, model, n):
    """
    Measure n features importance ratio for a specific model
    
    :param: df: classifier feature coefficients dataframe
    :param: model: model label
    :param: n: n first features to select
    
    :return: a dataframe
    """
    features = df['feature'][:n].tolist()
    fimp_ratio = round(df['coefficient'][:n].sum() / df['coefficient'].sum()*100, 2)
    return pd.DataFrame({f'{n} most important features': [features],
                         'model': [model],
                         'coefficient ratio (%)': [fimp_ratio]})