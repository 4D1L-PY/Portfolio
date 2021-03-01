from src.nlp.text_preprocessor import np, pd
from googletrans import Translator


def words_translator(text, language='en'):
    """
    Function which translate a word to another language (default translated language is english) by using
    Google Translate API
    :param text: a text (sentence(s) or word(s)) (str)
    :param language: destination language code
    (e.g. : https://py-googletrans.readthedocs.io/en/latest/#googletrans-languages)

    :return: a text translated
    """
    filtered_text = ' '.join(filter(str.isalpha, text.split()))
    try:
        translated_text = Translator().translate(filtered_text, dest=language).text
        if filtered_text == text:
            return translated_text
        return translated_text + ' ' + ''.join([w for w in text if w not in filtered_text])
    except AttributeError:
        return words_translator(text)


def autotag_question(vectorized_question, model, multilabel_encoder, upperize=False):
    """
    Predict a list of tags for a specific question

    :param vectorized_question: a vectorized question (numpy array)
    :param model: a trained text classifier model
    :param multilabel_encoder: a multi-label encoder instance
    :param upperize: apply uppercase function to predicted tags (boolean)

    :return: predicted tags labels (list of strings)
    """
    # Predict tags
    predicted_tags = np.round(model.predict(vectorized_question))
    # Get tag labels
    predicted_tags_labels = multilabel_encoder.inverse_transform(predicted_tags)
    predicted_tags_labels = list(set([tag for tags in predicted_tags_labels for tag in tags if len(tags) > 0]))
    if upperize:
        predicted_tags_labels = [tag.upper() for tag in predicted_tags_labels]
    return predicted_tags_labels


def compare_n_predicted_results(df, x_test, y_test, multilabel_encoder, n=10, text_col='Title'):
    """
    Compare predicted tags with ground truth from testing set

    :param df: a dataframe with question titles & body
    :param x_test: features from testing set
    :param y_test: targets from testing set
    :param multilabel_encoder: a multi-label encoder instance
    :param n: number of selected questions
    :param text_col: question part selected (title or body)
    """
    # Get n random questions (index) from testing set
    x_test_idx = np.random.randint(len(x_test), size=n)
    x_test_title = pd.DataFrame(x_test).join(df[text_col])[text_col]
    # For each testing questions display predicted tags & ground truth
    for i in x_test_idx:
        testing_question = 'Testing Question n°{0} :\n- {1}'.format(i, x_test_title.iloc[i])
        print(testing_question)
        print('-'*100)
        predicted_tags = autotag_question(x_test.iloc[i])[0]
        print(f'Predicted Tags : {predicted_tags}')
        ground_truth = multilabel_encoder.inverse_transform(np.array([y_test[i]]))[0]
        print(f'Ground Truth: {ground_truth}\n')
