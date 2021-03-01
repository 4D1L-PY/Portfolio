import keras.backend as K
from config.main_config import TEXT_CLS_METRICS
from src.nlp.text_preprocessor import pd


def evaluate_text_classifier(y_pred, y_test, metrics_dict=TEXT_CLS_METRICS, p=2):
    """
    Helper which evaluate a text classifier with specified metrics from sklearn

    :param y_pred: Estimated targets values as returned by a neural network classifier
    :param y_test: Ground truth target values
    :param metrics_dict: metrics dictionary with metric labels as keys and metric functions as values
    :param p: rounded metric decimal precision

    :return: a dataframe with results from each metrics
    """
    cls_results = {}
    for metric_label, metric_funct in metrics_dict.items():
        if metric_label is 'accuracy':
            cls_results[metric_label] = round(metric_funct(y_pred, y_test) * 100, p)
        elif metric_label is 'recall':
            cls_results[metric_label] = round(metric_funct(y_pred, y_test, average='samples', zero_division=0) * 100, p)
        else:
            cls_results[metric_label] = round(metric_funct(y_pred, y_test, average='samples') * 100, p)
    return pd.DataFrame(cls_results, index=['results'])


def keras_f1_score(y_true, y_pred):
    """
    Keras F1-score

    :param y_true: Ground truth target values
    :param y_pred: Estimated targets values as returned by a neural network classifier

    :return: F1-score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall)/(precision + recall + K.epsilon())
    return f1_val

