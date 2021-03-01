import os
import re
import tarfile
import pandas as pd
import numpy as np
from glob import glob
from googletrans import Translator

########################################################################################################################
#                                            Data cleaning functions                                                   #
########################################################################################################################


def get_unique_values(df, col, verbose=True):
    unique_values = df[col].unique().tolist()
    if verbose:
        print(f'Unique {col} : {len(unique_values)}')
    return unique_values


def delete_cols(df, variables, inplace=True):
    """
    Delete a list of variables (columns)
    :param df: a data frame
    :param variables: a list a columns labels
    :param inplace:
    """
    df.drop(columns=variables, inplace=inplace)
    return


def drop_duplicates_for_each_col(df, columns):
    """
    Drop duplicates values for each column in columns list

    :param df: dataframe
    :param columns: columns (labels) list
    """
    for column in columns:
        df.drop_duplicates(subset=column, inplace=True)


def drop_columns_with_few_data(df, threshold=0.4):
    """
    Drop columns which contains too many NaN values based on a specific threshold

    :param df:
    :param threshold:
    """
    # df[df.columns[df.isna().sum()/df.shape[0] < threshold]]
    for col in df.columns:
        if (df[df[col].isnull()].shape[0] / df.shape[0]) > threshold:
            df.drop(columns=col, inplace=True)


def nan_count(df, cols=None, precision=1, margin_size=100):
    def compute_nan_count(col=cols):
        NaN_values = len(df[df[col].isnull()])
        NaN_ratio = round((NaN_values / df.shape[0]) * 100, precision)
        NaN_count_as_str = "{}  ({}%)".format(NaN_values, NaN_ratio)
        return NaN_count_as_str
    if (cols is not None) and (type(cols) is str):
        print(compute_nan_count())
    else:
        cols = df.columns if cols is None else cols
        for col in cols:
            print(f'{col:-<{margin_size}} {compute_nan_count(col)}')

        
def display_unique_values_for_each_col(df, cols=None, margin_size=100, display_type='val', return_data=None):
    if cols is None:
        cols = df.columns
    elif cols is str and cols in df.columns:
        cols = [cols]
    total_unique_values = []
    for col in cols:
        unique_values = get_unique_values(df, col, verbose=False)
        total_unique_values.extend(unique_values)
        if display_type is 'val':
            results = unique_values
        elif display_type is 'ct':
            results = len(unique_values)
        else:
            raise Exception(f'{display_type} is invalid')
        print(f'{col:-<{margin_size}} {results}')
    if return_data is 'val':
        return total_unique_values
    elif return_data is 'ct':
        return len(total_unique_values)
    

def filter_variable_values(df, variable, filter_type, threshold):
    if filter_type is 'l':
        return df[df[variable] < threshold]
    elif filter_type is 'le':
        return df[df[variable] <= threshold]
    elif filter_type is 'g':
        return df[df[variable] > threshold]
    elif filter_type is 'ge':
        return df[df[variable] >= threshold]
    elif filter_type is 'eq':
        return df[df[variable] == threshold]
    elif filter_type is 'neq':
        return df[df[variable] != threshold]


def translate_col_values_by_regex(df, col, terms_translator=None):
    """
    Method which map a dictionary to a specific dataframe column by using regex

    :param df: a dataframe
    :param col: a column which will be mapped
    :param terms_translator: a translation dictionary
    """
    terms_keys, terms_values = list(terms_translator.keys()), list(terms_translator.values())
    df[col].replace(to_replace=terms_keys, value=terms_values, regex=True, inplace=True)


def find_non_common_null_values(df, col1, col2):
    return df[[col1, col2]][(df[col1].isnull()) & (df[col2].isnull() is False)]


def get_data_loss(ct1, ct2, axis=0, precision=1):
    axis_label = 'rows' if axis == 0 else 'columns'
    ct1, ct2 = [ct.shape[axis] if isinstance(ct, pd.DataFrame) else ct for ct in [ct1, ct2]]
    loss = round(100 - ((ct1 / ct2) * 100), precision)
    if loss > 0:
        print("Dataframe {} reduced by {} % ({} -> {})".format(axis_label, loss, ct2, ct1))
    elif loss < 0:
        print("Dataframe {} increased by {} % ({} -> {})".format(axis_label, abs(loss), ct2, ct1))
    else:
        print("Dataframe {} inchanged ({} -> {})".format(axis_label, ct2, ct1))


def get_index_from_features(df, features_labels):
    """
    Extract features index list from a dataframe based on features labels
    """
    df_t = df.T.copy().reset_index()
    features_idx = df_t[df_t['index'].isin(features_labels)].index
    return features_idx.tolist()


def get_dummy_indicators(df, variable, drop=True, prefix_origin_col=False):
    df_origin = df.copy()
    df_indicators = pd.get_dummies(df_origin[variable], prefix=variable if prefix_origin_col is True else None)
    df_destination = pd.concat([df_origin, df_indicators], axis=1)
    if drop:
        delete_cols(df_destination, variable)
    return df_destination


def get_centroids_from_categories(df, category_label):
    dfc = df.copy()
    categories_labels = dfc[category_label].unique().tolist()
    centroids = []
    # Select quantitatives variables
    quanti_cols = df.select_dtypes(include=[np.number]).columns
    for col in quanti_cols:
        centroids_col = dfc.groupby(category_label)[col].apply(lambda x: np.median(x.tolist(), axis=0)).values
        centroids.append(centroids_col.tolist())
    df_centroids = pd.DataFrame({col: c for col, c in zip(quanti_cols, centroids)}, index=categories_labels)
    df_centroids.index.name = category_label
    return df_centroids


def filter_underpopulated_classes(df, target, n_members=1):
    """
    Filter underpopulated classes from a specific dataframe

    :param df: a dataframe
    :param target: target column label
    :param n_members: minimal number of members to apply classes filtering

    :return: a filtered dataframe
    """
    # Get unique classes (if multi-label corresponds to all availables combinations)
    classes, y_indices = np.unique(df[target], return_inverse=True)
    # Get classes distribution
    classes_dist = np.bincount(y_indices)
    # Get underpopulated classes indices (classes which have only 1 member)
    underpopulated_classes_indices = np.where(classes_dist == n_members)[0]
    # Get underpopulated classes
    underpopulated_classes = classes[underpopulated_classes_indices]
    # Filter underpopulated classes
    df = df[~df[target].map(lambda t: any(cls for cls in underpopulated_classes if cls == t))]
    return df


########################################################################################################################
#                                                Utilities functions                                                   #
########################################################################################################################


def make_tarfile(filename, path, extension=''):
	"""
	Make .tar file

	:param filename: filename (str)
	:param path: relative path to .tar file
	:param extension: add gz extension (default extension is .tar)

	"""
    with tarfile.open(filename, f"w:{extension}") as tar:
	tar.add(path)
        # tar.add(path, arcname=os.path.basename(path))


def uncompress_tarfile(file_path):
	"""
	Uncompress .tar file

	:param file_path: relative path to .tar file
	"""
	with tarfile.open(file_path, "r:") as files:
    	files.extractall()



def concat_samples(sample_path='data/csv/',
                   sample_frac=None,
                   sort=True,
                   n_first_samples=None,
                   n_last_samples=None):
    """
    Merge multiple csv files

    :param sample_path: relative path to sample data
    :param sample_frac: (Optional), ratio which sample merged dataframes
    :param sort: boolean which enable/disable sorting samples by year
    :param n_first_samples: n first selected samples
    :param n_last_samples: n last selected samples

    :return: a dataframe
    """
    files = glob(f'{sample_path}*.csv')
    if sort:
        files.sort(key=lambda x: re.compile(r"\d+").findall(x)[0])
    selected_samples = files[:n_first_samples] if n_last_samples is None else files[-n_last_samples:]
    df = pd.concat([pd.read_csv(file, low_memory=False) for file in selected_samples])
    if sample_frac is not None:
        df = df.sample(frac=sample_frac)
    return df


def get_memory_usage(df):
    """
    Get data frame memory usage
    :param df: a dataframe
    :return: a readable memory usage (with unit label)
    """
    if type(df) in [int, float]:
        memory_as_octet = df
    else:
        memory_as_octet = df.memory_usage(deep=True).sum()
    # Get the order of magnitude of the memory number
    order_magnitude_memory = len(str(memory_as_octet)) - 1
    unit_dict = {'o': 0, 'Ko': 3, 'Mo': 6, 'Go': 9, 'To': 12}
    unit_labels, unit_orders = list(unit_dict.keys()), list(unit_dict.values())
    # Loop on dictionary items in order to match a converted memory with a unit label
    for unit_label, order_magnitude in unit_dict.items():
        if order_magnitude_memory == order_magnitude:
            memory_converted = memory_as_octet / pow(10, order_magnitude)
            return '{} {}'.format(memory_converted, unit_label)
        else:
            if order_magnitude_memory not in unit_orders:
                hightest_order = max([x for x in unit_orders if order_magnitude_memory > x])
                # decrement by the difference between the memory order magnitude and the hightest order
                order_magnitude_memory -= (order_magnitude_memory - hightest_order)
            memory_converted = memory_as_octet / pow(10, order_magnitude_memory)
            return '{} {}'.format(memory_converted, unit_labels[unit_orders.index(order_magnitude_memory)])

        
def data_info(df):
    """
    :param df: a dataframe
    :return: data type & memory usage information (dataframe)
    """
    dtype_data = df.dtypes.value_counts()
    dtype_info = ", ".join([f'{dtype}({ct})' for dtype, ct in dtype_data.iteritems()])
    mem_info = get_memory_usage(df)
    df_info = pd.DataFrame({'Data type': [dtype_info], 'Memory usage': mem_info}, index=['Results'])
    return df_info

        
def check_low_cardinality(df, categorical_col):
    """
    Check if a categorical variable as low cardinality (= the amount of unique values is lower 
    than 50% of the count of these values)
    
    :param df: a dataframe
    :param categorical_col: a categorical column label
    :return: boolean value
    """
    low_cardinality_ratio = (len(df[categorical_col].unique())/len(df[categorical_col]))
    return True if low_cardinality_ratio < 0.5 else False


def optimize_data_types(df):
    """
    Optimize data types for each columns labels, by reducing the size of dataframe
    cf:
    https://medium.com/@vincentteyssier/optimizing-the-size-of-a-pandas-dataframe-for-low-memory-environment-5f07db3d72e
    """
    for col in df.columns:
        dtype_extracted = ''.join(c for c in df[col].dtype.name if c.isdigit() is False)
        if dtype_extracted == 'int':
            df.loc[:, col] = pd.to_numeric(df[col], downcast='integer')
        elif dtype_extracted == 'float':
            df.loc[:, col] = pd.to_numeric(df[col], downcast='float')
        elif dtype_extracted == 'object':
            low_cardinality_cond = check_low_cardinality(df, col)
            if low_cardinality_cond:
                df.loc[:, col] = df[col].astype('category')


def read_number_cleaner(number):
    """
    Improve number readability
    :param number: int or float number
    :return: readable number as str
    """
    if type(number) is float:
        int_part, float_part = str(number).split('.')
        return read_int_cleaner(int_part) + '.' + float_part
    else:
        return read_int_cleaner(number)


def read_int_cleaner(number):
    """
    Improve int number readability
    (Example: 2500 -> '2 500')

    :param number: int number
    :return: readable int number as str
    """
    str_number = str(number)
    str_number, frag_number = str_number[:-3], str_number[-3:]
    if len(str_number) == 0:
        return frag_number
    else:
        return read_int_cleaner(str_number) + ' ' + frag_number

    
def format_run_time(run_time_value):
    """

    :param run_time_value: (str or timedelta object)
    :return:
    """
    time_str = str(run_time_value)
    time_labels, time_values = ['h', 'min', 's'], time_str.split(':')
    time_dict = {time_label: round(float(time_val)) for time_label, time_val in zip(time_labels, time_values)}
    run_time_str = ' '.join(['{} {}'.format(time_val, k) for k, time_val in time_dict.items() if time_val != 0])
    return run_time_str


def convert_str_time_to_time(df, col='Run time', time_unit='s'):
    """
    e.g : "1 min 10s" --> "00:01:10"
    """
    df[col] = pd.to_timedelta(df[col], unit=time_unit).dt.seconds
    df[col] = pd.to_datetime(df[col], unit=time_unit).dt.time


def words_translator(word, language='en'):
    """
    Function which translate a word to another language (default translated language is english) by using
    Google Translate API
    :param word: a word (str)
    :param language: destination language code
    (cf : https://py-googletrans.readthedocs.io/en/latest/#googletrans-languages)
    :return: a word translated
    """
    return Translator().translate(word, dest=language).text
