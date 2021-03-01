from tensorflow import keras
from src.text_classifier_evaluator import pickle_data, keras_f1_score


########################################################################################################################
#                                          TEXT NORMALIZER PARAMETERS                                                  #
########################################################################################################################

# Text normalizer parameters

TEXT_NORMALIZER_PARAMS = {'no_digits': True,
                          'no_punctuation': True,
                          'no_repeated_characters': True,
                          'no_single_letters': True,
                          'no_stopwords': True,
                          'stopwords_params': {'lib': 'spacy', 'min_token_length': 2},
                          # Keep specific programming tags like 'c++', 'c#' etc ...
                          'domain_words': pickle_data(filename='tags', folder='data/preprocessing', method='r')
                          }


########################################################################################################################
#                                           PREPROCESSING PARAMETERS                                                   #
########################################################################################################################

# Load multi label binarizer
multilabel_encoder = pickle_data(filename='multilabel_encoder', folder='data/preprocessing', method='r')
# Load keras tokenizer
gru_tokenizer = pickle_data(filename='GRU_tokenizer', folder='data/preprocessing', method='r')
# Load GRU trained model
gru = keras.models.load_model('data/models/stackoverflow_tag_predictor.h5', custom_objects={'f1_score': keras_f1_score})


########################################################################################################################
#                                               APP PARAMETERS                                                         #
########################################################################################################################

# App logo filename
APP_LOGO_NAME = 'stackoverflow_autotagger_logo.svg'

