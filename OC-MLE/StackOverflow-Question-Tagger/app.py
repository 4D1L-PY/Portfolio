# import flask_monitoringdashboard as dashboard
from flask import Flask
from flask import request, render_template, session, redirect, url_for, jsonify, send_file
# from scipy.sparse import hstack
from src.nlp.text_preprocessor import *
from src.nlp.text_classifier_evaluator import keras_f1_score
from config.main_config import TEXT_NORMALIZER_PARAMS
from utils.autotagger import words_translator, autotag_question


########################################################################################################################
#                                               PREPROCESSING                                                          #
########################################################################################################################


# Main app configuration

app = Flask(__name__, template_folder='templates', static_folder='static')
# Add Flask monitoring dashboard
# dashboard.bind(app)
# App logo filename
APP_LOGO_NAME = 'stackoverflow_autotagger_logo.svg'

# Load preprocessing data

# Keep specific programming tags like 'c++', 'c#' etc ...
TEXT_NORMALIZER_PARAMS['domain_words'] = pickle_data(filename='tags', folder='data/preprocessing', method='r')
# Load multi label binarizer
multilabel_encoder = pickle_data(filename='multilabel_encoder', folder='data/preprocessing', method='r')
# Load keras tokenizer
gru_tokenizer = pickle_data(filename='GRU_tokenizer', folder='data/preprocessing', method='r')
# Load GRU trained model
gru = keras.models.load_model('data/models/stackoverflow_tag_predictor.h5', custom_objects={'f1_score': keras_f1_score})


########################################################################################################################
#                                               MAIN VIEW                                                              #
########################################################################################################################


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    API landing page

    :return: API main view (landing page)
    """
    return render_template('index.html', image_name=APP_LOGO_NAME)


########################################################################################################################
#                                                 TASKS                                                                #
########################################################################################################################


@app.route('/_tag_question', methods=['POST'])
def tag_question():
    """
    Question tagger feature which implies :
        - Retrieve question from user form
        - Text preprocessing (filter HTML tags, translate to english, tokenize, normalize question)
        - Auto-tag question with deep neural network model (GRU)

    :return: question predicted tags (JSON)
    """
    question = request.form['text']
    # Filter HTML tags
    question = filter_html_tags(question)
    # Translate a question from an unknown language to english
    question = words_translator(question)
    # Lowerize & Tokenize question
    tokenized_question = tokenize(question, lowerize=True)
    # Normalise tokenized question
    normalized_tokens = text_normalizer(tokenized_question, **TEXT_NORMALIZER_PARAMS)
    # GRU preprocessing (preprocess normalized tokens)
    padded_question = keras.preprocessing.sequence.pad_sequences(gru_tokenizer.texts_to_sequences(normalized_tokens),
                                                                 maxlen=gru.get_layer('embedding').input_length)
    # GRU Predicted tags
    gru_predicted_tags = autotag_question(padded_question, gru, multilabel_encoder)
    predicted_tags_by_model = {'GRU': gru_predicted_tags}
    return jsonify(predicted_tags_by_model)


if __name__ == '__main__':
    app.run(debug=True)
