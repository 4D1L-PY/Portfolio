import keras.backend as K
import pickle


# Keras F1 score
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


# Pickle wrapper
def pickle_data(filename, data=None, folder=None, method='w'):
    """
    Pickle data

    :param filename: a filename (str)
    :param data: data to save as .pkl file
    :param folder: folder name
    :param method: pickling method (w -> write file, r -> read file)

    :return: python object from .pkl file (if method is 'r', else return nothing)
    """
    filename = filename if folder is None else f'{folder}/{filename}'
    if method is 'w':
        with open(f'{filename}.pkl', 'wb') as file:
            pickle.dump(data, file)
    elif method is 'r':
        with open(f'{filename}.pkl', 'rb') as file:
            file_data = pickle.load(file)
        return file_data