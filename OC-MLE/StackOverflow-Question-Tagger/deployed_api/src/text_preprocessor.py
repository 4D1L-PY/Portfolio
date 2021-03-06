import itertools
import string
import re
import pandas as pd
import spacy
import nltk
from nltk.stem.snowball import SnowballStemmer
from fuzzywuzzy import fuzz, process
from bs4 import BeautifulSoup
from requests import get


########################################################################################################################
#                                          SPACY CONFIGURATION                                                         #
########################################################################################################################


class SpacyConfigurator:

    def __init__(self, language, tags_url='https://spacy.io/api/annotation'):
        self.language = language
        self.tags_url = tags_url
        self.pipeline_tags = ['tagger',
                              'parser',
                              'ner',
                              'textcat',
                              'entity_linker',
                              'entity_ruler',
                              'sentencizer',
                              'merge_noun_chunks',
                              'merge_entities',
                              'merge_subtokens']
        self.tags = self.get_tags()
        # To do : add more tag types

    def get_tags(self):
        html_soup = BeautifulSoup(get(self.tags_url).text, 'html.parser')
        # Tagging type titles
        title_parser = html_soup.find_all('a', {"class": "heading-text e80ba60d"})[4:-3]
        tagging_type_titles = [title.text.strip() for title in title_parser]
        # Tagging type subtitles
        subtitle_patterns = ['Universal', 'English', 'German']
        subtitle_parser = html_soup.find_all('span', {"class": "heading-text"})
        subtitles = [t.text for t in subtitle_parser for s in subtitle_patterns if s in t.text]
        # Tagging tables
        tagging_tables_parser = html_soup.find_all('table', {"class": "_59fbd182"})
        tagging_tables = [pos_table for pos_table in tagging_tables_parser]
        # Select relevant titles
        rebuilt_titles = subtitles + tagging_type_titles[2:]
        # Tags html tag class pattern
        tags_cls_pattern = {"class": "_1d7c6046"}
        spacy_tags = {}
        for i, title in enumerate(rebuilt_titles):
            tags = [t.text for t in tagging_tables[i].find_all('code', tags_cls_pattern) if '=' not in t.text]
            tags = list(set(tags))
            if i in [1, 2]:
                spacy_tags[f'{title} Part-of-speech Tags'] = tags
            elif i in [4, 5]:
                spacy_tags[f'{title} Dependency Labels'] = tags
            else:
                spacy_tags[title] = tags
        return spacy_tags
    
    
# Language
LANG = 'en_core_web_sm'  # en

spacy_conf = SpacyConfigurator(LANG)

SPACY_UNIVERSAL_POS_TAGS = spacy_conf.tags['Universal Part-of-speech Tags']

SPACY_PIPELINE_TAGS = spacy_conf.pipeline_tags

SPACY_DEFAULT_STOPWORDS = spacy.load(LANG).Defaults.stop_words

# Text normalizer instances

LEMMATIZER_INSTANCE = spacy.load(LANG, disable=['parser', 'ner'])

POS_TAGGER_INSTANCE = spacy.load(LANG, disable=SPACY_PIPELINE_TAGS[1:])

NER_INSTANCE = spacy.load(LANG, disable=['tagger',
                                         'parser',
                                         'textcat',
                                         'sentencizer',
                                         'merge_noun_chunks',
                                         'merge_subtokens'])    


########################################################################################################################
#                                          TEXT PREPROCESSING                                                          #
########################################################################################################################


def replace_characters(sentence, translator):
    """
    Replace multiple characters by dictionary pattern

    :param sentence: a string
    :param translator: a translator table (dict = {old character: new character})

    :return: a string with replaced characters
    """
    chars_to_replace = translator.keys()
    chars_to_match = '({})'.format('|'.join(map(re.escape, chars_to_replace)))
    cleaned_sentence = re.sub(chars_to_match, lambda m: translator[m.group()], sentence)
    return cleaned_sentence


def filter_repeated_characters(text, thr=3):
    """
    Filter repeated characters from text data

    :param text: a string or list of tokens
    :param thr: repeated characters threshold (maximum repeated character occurrence)

    :return: tokens or sentence without repeated characters
    """
    text_to_filter = [text] if type(text) is str else text
    text_filtered = []
    for word in text_to_filter:
        groups = itertools.groupby(word)
        consecutive_characters_count = [sum(1 for _ in group) for char, group in groups]
        if max(consecutive_characters_count) < thr:
            text_filtered.append(word)
    if type(text) is str:
        return ''.join(text_filtered)
    else:
        return text_filtered


def filter_single_letters(tokens):
    """
    Filter word-letters from text data

    :param tokens: a list of tokens

    :return: tokens without word-letters
    """
    filtered_tokens = [token for token in tokens if token not in string.ascii_letters]
    return filtered_tokens


def filter_punctuation(text, regex_pattern='[^0-9a-zA-Z]+'):
    """
    Filter punctuation characters from text data

    :param text: a string or list of tokens
    :param regex_pattern: regex pattern to filter punctuation (default is non-alphanumeric characters)

    :return: a token or sentence without punctuation
    """
    # Manage text data type (a list of tokens or a sentence)
    tokens = []
    for token in (text if type(text) is list else text.split()):
        # Filter punctuation and tokens which are letters
        if token not in string.ascii_letters:
            cleaned_token = re.sub(regex_pattern, ' ', token).strip()
            # Filter empty and space tokens
            if any(char for char in cleaned_token if char not in ['', ' ']):
                tokens.append(cleaned_token)
    # Flatten n-grams tokens ['e g'] -> ['e', 'g']
    tokens = [subtoken if ' ' in token else token for token in tokens for subtoken in token.split()]
    return tokens if type(text) is list else ' '.join(tokens).strip()


def filter_digits(text):
    """
    Filter digits from text data

    :param text: a string (token or sentence)

    :return: a token or sentence without digits
    """
    # Manage text data type (a list of tokens or a sentence)
    if type(text) is list:
        tokens_with_no_digits = []
        for token in text:
            # Remove digits in text data (e.g: 'a1' -> 'a')
            token_with_no_digits = ''.join([char for char in token if not char.isdigit()])
            if any(char for char in token_with_no_digits if char not in ['', ' ']):
                tokens_with_no_digits.append(token_with_no_digits)
    else:
        tokens_with_no_digits = ''.join([char for char in text if not char.isdigit()]).strip()
    return tokens_with_no_digits


def filter_html_tags(text):
    """
    Filter HTML tags from text

    :param text: a string with HTML tags

    :return: a string without HTML tags
    """
    text_with_no_html_tags = BeautifulSoup(text, 'html.parser').get_text()
    return text_with_no_html_tags


def filter_stopwords(text, other_stopwords=None, lang='english', min_token_length=0, lib='nltk'):
    """
    Filter stopwords from text data (tokens or sentence)
    (see :
    https://medium.com/towards-artificial-intelligence/stop-the-stopwords-using-different-python-libraries-ffa6df941653)

    :param tokens: list of tokens
    :param other_stopwords: secondary list which contains other stopwords
                            (will be merged with original stopwords list)
    :param lang: stopwords list language
    :param min_token_length: minimal token length

    :return: a filtered list of tokens
    """
    # Get stopwords list by language
    if lib is 'nltk':
        stopwords_list = nltk.corpus.stopwords.words(lang)
    elif lib is 'spacy':
        stopwords_list = SPACY_DEFAULT_STOPWORDS
    # TO DO : add gensim & sklearn lists
    if other_stopwords is not None:
        stopwords_list.extend(other_stopwords)
    if type(text) is str:
        text = tokenize(text)
    filtered_tokens = [token for token in text if (token not in stopwords_list) \
                       & (len(token) > min_token_length)]
    # Manage text data type (a list of tokens or a sentence)
    return filtered_tokens if type(text) is list else ' '.join(filtered_tokens)  # .strip()


def filter_words_by_levenshtein_similarity(words, threshold=0.50, metric=fuzz.token_set_ratio):
    """
    Filter list of words based on levenshtein similarity

    :param words: list of words
    :param threshold: levenshtein similarity threshold ratio
    :param metric: metric used in order to compute levenshtein similarity

    :return: list of words filtered by levenshtein similarity
    """
    similar_words = []
    # First iteration over words list
    for outer_index, word in enumerate(words):
        # Second iteration over words list
        for inner_index in range(len(words)):
            # Compute levenshtein similarity ratio (which depends on used metric)
            levenshtein_similarity = metric(word, words[inner_index]) * 0.01
            # Find similar pair of words based on levenshtein similarity threshold ratio
            if (inner_index != outer_index) and (levenshtein_similarity > threshold):
                similar_pair_of_words = [word, words[inner_index]]
                # Check if similar pair of words are not already in words list
                if all(s for s in similar_pair_of_words if s not in words):
                    similar_words.extend(similar_pair_of_words)
    # Filter similar words from origin words list
    return [w for w in words if w not in similar_words]


def filter_most_common_tokens(tokens, n=None):
    """
    Filter n most common tokens

    :param tokens: list of tokens
    :param n: n most common selected tokens (float or int value)

    :return: a filtered list of tokens
    """
    tokens_freqdist = nltk.FreqDist(tokens)
    # Filter uniform distribution len(set(tokens_freqdist.values()))
    if len(tokens) > 10:
        n_most_common_tokens = int(round(n * len(tokens))) if type(n) is float else n
        most_freq_tokens = list(list(zip(*tokens_freqdist.most_common(n_most_common_tokens)))[0])
        tokens_filtered = [token for token in tokens if token not in most_freq_tokens]
        return tokens_filtered
    return tokens


def filter_tokens_by_length(tokens, min_cond, max_cond, keep_domain_based_words=False, domain_based_words=None):
    """

    """
    if keep_domain_based_words:
        domain_based_tokens = [token for token in tokens if token in domain_based_words]
    filtered_tokens = []
    for token in tokens:
        if ((len(token) > min_cond) and (len(token) < max_cond)) and (token not in domain_based_words):
            filtered_tokens.append(token)
        elif (keep_domain_based_words is True) and (token in domain_based_tokens):
            filtered_tokens.append(token)
    return filtered_tokens


def filter_questions_by_token_count(df, tokens_col, min_cond, max_cond):
    """

    """
    df = df[(df[tokens_col].map(len) > min_cond) & (df[tokens_col].map(len) < max_cond)]
    return df


def get_domain_based_tokens(tokens, domain_based_words):
    """
    Extract domain-based words from tokens

    :param tokens: list of tokens
    :param domain_based_words: list of domain-based words

    :return: a filtered list of tokens & domain-based tokens
    """
    domain_based_tokens = [token for token in tokens if token in domain_based_words]
    tokens = [token for token in tokens if token not in domain_based_tokens]
    return domain_based_tokens, tokens


def tokenize(sentence, lowerize=False):
    """
    Tokenize a sentence

    :param sentence: text data
    :param lowerize: boolean which enable/disable lowerizing text data

    :return: a list of tokens
    """
    tokens = nltk.word_tokenize(sentence.lower() if lowerize else sentence)
    return tokens


def stemmerize(data, lang='english'):
    """
    Stemmerize a list of tokens or a sentence

    :param data: list of tokens (or a string)
    :param lang: stemmer language

    :return: a stemmerized list of tokens (or a string)
    """
    stemmer = SnowballStemmer(lang)
    tokens_cond = (type(data) is list)
    stemmerized_tokens = [stemmer.stem(token) for token in data] if tokens_cond else stemmer.stem(data)
    return stemmerized_tokens


def lemmatize(data, nlp_instance=LEMMATIZER_INSTANCE, pos_tags_kept=SPACY_UNIVERSAL_POS_TAGS, keep_original_text=False):
    """
    Lemmatize a list of tokens or a sentence

    :param data: list of tokens (or a string)
    :param nlp_instance: spacy loaded instance
    :param pos_tags_kept: Spacy tags to filter
    :param keep_original_text: boolean which enable/disable adding original token/word to lemmatized list of tokens

    :return: a lemmatized list of tokens (or a string)
    """
    data_type = type(data)
    data = [data] if data_type is str else data
    lemmatizer = nlp_instance
    results = list(lemmatizer.pipe(data))
    lemmatized_tokens = []
    for document in results:
        for token in document:
            # Try to extract token's lemma based on specific POS tags kept
            if token.pos_ in pos_tags_kept:
                lemmatized_tokens.append(token.lemma_)
            # Otherwise keep original token text or ignore token (which is filtered from lemmatized tokens)
            elif keep_original_text:
                lemmatized_tokens.append(token.text)
    return lemmatized_tokens if data_type is list else ' '.join(lemmatized_tokens)


def extract_n_grams(data, n=2):
    """
    Extract n-grams from text data

    :param data: list of tokens (or a string)
    :param n: n-grams considered (default is bigrams)

    :return: n-grams list
    """
    data = data if type(data) is list else tokenize(data)
    ngrams = list(nltk.ngrams(data, n))
    return ngrams


def pos_tag(data, nlp_instance=POS_TAGGER_INSTANCE, filtered_tags=(), only_tokens=False, module_type='spacy'):
    """
    POS tag a list of tokens (or a sentence)

    :param data: list of tokens (or a string)
    :param nlp_instance: spacy loaded instance
    :param filtered_tags: POS tags to filter
    :param only_tokens: boolean which enable/disable returning only filtered tokens
    :param module_type: module architecture type ('spacy' or 'nltk')

    :return: tagged tokens
    """
    if module_type is 'spacy':
        pos = nlp_instance
        if type(data) is list:
            results = list(pos.pipe(data))
            tagged_tokens = [(tk.text, tk.tag_) for doc in results for tk in doc if tk.tag_ not in filtered_tags]
        else:
            results = pos(data)
            tagged_tokens = [(doc.text, doc.tag_) for doc in results if doc.tag_ not in filtered_tags]
    else:
        tagged_tokens = nltk.pos_tag(data) if type(data) is list else nltk.pos_tag(tokenize(data))
    if only_tokens:
        tagged_tokens = [token for token, tag in tagged_tokens if tag not in filtered_tags]
    return tagged_tokens


def name_entity_recognize(data, nlp_instance=NER_INSTANCE, filtered_entities=(), only_tokens=False):
    """
    Extract name entities from a list of tokens (or a sentence)

    :param data: list of tokens (or a string)
    :param nlp_instance: spacy loaded instance
    :param filtered_entities: entities to filter
    :param only_tokens: boolean which filter entities from NER results

    :return: tokens entities
    """
    ner = nlp_instance
    if type(data) is list:
        results = list(ner.pipe(data))
        tokens_ents = [(ent.text, ent.label_) for doc in results
                       for ent in doc.ents if ent.label_ not in filtered_entities]
    else:
        results = ner(data)
        tokens_ents = [(ent.text, ent.label_) for ent in results.ents]
    if only_tokens:
        tokens_ents = [token for token, entity in tokens_ents]
    return tokens_ents


def text_normalizer(text,
                    lowerizer=False,
                    no_digits=False,
                    no_punctuation=False,
                    no_repeated_characters=False,
                    no_single_letters=False,
                    no_stopwords=False,
                    no_duplicates=False,
                    no_similar_words=False,
                    filter_most_common_tokens=False,
                    stemmerizer=False,
                    lemmatizer=False,
                    pos_tagger=False,
                    stopwords_params=None,
                    levenshtein_params=None,
                    most_common_tokens_filter_params=None,
                    stemmerizer_params=None,
                    lemmatizer_params=None,
                    pos_tagger_params=None,
                    duplicates_type='token',
                    domain_words=()):
    """
    Normalize a sentence

    :param text: text data

    :param lowerizer: boolean which enable/disable lowerizing text data
    :param no_digits: boolean which enable/disable cleaning & filtering string which contains digits
    :param no_punctuation: boolean which enable/disable filtering punctuation
    :param no_repeated_characters: boolean which enable/disable filtering repeated characters in words
    :param no_single_letters: boolean which enable/disable filtering words-letters

    :param no_stopwords: boolean which enable/disable filtering stopwords in string
    :param no_duplicates: boolean which enable/disable filtering non domain based duplicates words
    :param no_similar_words: boolean which enable/disable filtering non domain based similar words
    :param filter_most_common_tokens: boolean which enable/disable filtering most common tokens

    :param stemmerizer: boolean which enable/disable stemming tokens
    :param lemmatizer: boolean which enable/disable lemmatize tokens
    :param pos_tagger: boolean which enable/disable POS tagging tokens

    :param stopwords_params: stopwords parameters (dict)
    :param levenshtein_params: levenshtein similarity parameters (dict)
    :param most_common_tokens_filter_params: filter most common tokens parameters (dict)
    :param stemmerizer_params: stemmerizer parameters (dict)
    :param lemmatizer_params: lemmatizer parameters (dict)
    :param pos_tagger_params: POS tagger parameters (dict)

    :param duplicates_type: duplicates type to filter, could be :

           - 'token': only concerns non domain based tokens
           - 'domain': only concerns domain based tokens
           - 'all': concerns non domain & domain based tokens

    :param domain_words: list of domain based words need to be kept

    :return: a list of tokens
    """

    # Tokenize text (if text is not a list of tokens)
    tokens = tokenize(text, lowerize=lowerizer) if type(text) is str else text
    # Get domain based tokens
    domain_based_tokens = []
    if len(domain_words) > 0:
        domain_based_tokens, tokens = get_domain_based_tokens(tokens, domain_words)
    # Remove digits in words and digits as str
    if no_digits:
        tokens = filter_digits(tokens)
    # Filter punctuation
    if no_punctuation:
        tokens = filter_punctuation(tokens)
    # Filter words with repeated characters
    if no_repeated_characters:
        tokens = filter_repeated_characters(tokens)
    # Filter words-letters ('a', 'b', ...)
    if no_single_letters:
        tokens = filter_single_letters(tokens)
    # Filter stopwords
    if no_stopwords:
        if stopwords_params is not None:
            tokens = filter_stopwords(tokens, **stopwords_params)
        else:
            tokens = filter_stopwords(tokens)
    # Filter duplicates
    if no_duplicates:
        if duplicates_type in ['token', 'all']:
            tokens = pd.Series(tokens, dtype='object').drop_duplicates(keep=False).tolist()
        if duplicates_type in ['domain', 'all']:
            domain_based_tokens = list(set(domain_based_tokens))
    # Filter similar words
    if no_similar_words:
        if levenshtein_params is not None:
            tokens = filter_words_by_levenshtein_similarity(tokens, **levenshtein_params)
        else:
            tokens = filter_words_by_levenshtein_similarity(tokens)
    # Filter most common tokens
    if filter_most_common_tokens:
        if most_common_tokens_filter_params is not None:
            tokens = filter_most_common_tokens(tokens, **most_common_tokens_filter_params)
        else:
            tokens = filter_most_common_tokens(tokens)
    # Enable stemmerizer
    if stemmerizer:
        if stemmerizer_params is not None:
            tokens = stemmerize(tokens, **stemmerizer_params)
        else:
            tokens = stemmerize(tokens)
    # Enable lemmatizer
    if lemmatizer:
        if lemmatizer_params is not None:
            tokens = lemmatize(tokens, **lemmatizer_params)
        else:
            tokens = lemmatize(tokens)
    # Enable POS tagging
    if pos_tagger:
        if pos_tagger_params is not None:
            tokens = pos_tag(tokens, **pos_tagger_params)
        else:
            tokens = pos_tag(tokens)
    # Merge generic tokens and domain based tokens
    tokens.extend(domain_based_tokens)
    return tokens

