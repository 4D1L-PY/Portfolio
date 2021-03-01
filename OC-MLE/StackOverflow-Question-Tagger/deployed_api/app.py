from flask import Flask
from flask import request, render_template, jsonify
from config.app_config import APP_LOGO_NAME, TEXT_NORMALIZER_PARAMS, multilabel_encoder, gru_tokenizer, gru, keras
from src.text_preprocessor import filter_html_tags, tokenize, text_normalizer
from utils.autotagger import words_translator, autotag_question


########################################################################################################################
#                                            APP PREPROCESSING                                                         #
########################################################################################################################

# Main app configuration
app = Flask(__name__, template_folder='templates', static_folder='static')


########################################################################################################################
#                                               MAIN VIEW                                                              #
########################################################################################################################


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    API landing page

    :return: main API view (landing page)
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
        - Text preprocessing (filter HTML tags, tokenize, normalize question)
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


########################################################################################################################
#                                                  RUN APP                                                             #
########################################################################################################################

if __name__ == '__main__':
    app.run(debug=True)
