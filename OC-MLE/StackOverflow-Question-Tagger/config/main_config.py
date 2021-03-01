import os
import spacy
from src.nlp.spacy_configurator import SpacyConfigurator
from src.evaluator import accuracy_score, precision_score, recall_score, f1_score, jaccard_score


########################################################################################################################
#                              STACK EXCHANGE DATA EXPLORER EXTRACTED DATA PATHS                                       #
########################################################################################################################

SO_2019_DATA_PATH = '../data/csv/data_2019/'

"""
Extraction method : sample data by month & filtered by Score & AnswerCount (repeat query for each year)

# Query type with SQL-SERVER

SELECT -- Question part
       q.Id,
       q.Score,
       q.AnswerCount,
       q.CreationDate,
       q.Title,
       q.Body,
       q.Tags,
       -- Accepted answer part
       a.Id AS AnswerId,
       a.Score AS AnswerScore,
       a.CreationDate AS AnswerCreationDate,
       a.Body AS Answer
FROM posts q
INNER JOIN posts a ON q.AcceptedAnswerId = a.Id
WHERE YEAR(q.CreationDate) = 2019
  AND MONTH(q.CreationDate) = 01 -- Repeat query for each month
  AND q.DeletionDate IS NULL     -- Valid Question (not deleted)
  AND q.Score > 0                -- Question with more upvotes than downvotes  
  AND q.AnswerCount > 0          -- Question with at least one answer
ORDER BY q.Id DESC

"""


########################################################################################################################
#                                          SPACY CONFIGURATION                                                         #
########################################################################################################################


# Language
LANG = 'en_core_web_sm' # en

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
#                                          TEXT NORMALIZER PARAMETERS                                                  #
########################################################################################################################


# Text normalizer parameters

TEXT_NORMALIZER_PARAMS = {'no_digits': True,
                          'no_punctuation': True,
                          'no_repeated_characters': True,
                          'no_single_letters': True,
                          'no_stopwords': True,
                          'stopwords_params': {'lib': 'spacy', 'min_token_length': 2},
                          'domain_words': None
                          }


########################################################################################################################
#                                       TEXT CLASSIFIER EVALUATION PARAMETERS                                          #
########################################################################################################################


# Model evaluation metrics
TEXT_CLS_METRICS = {'accuracy': accuracy_score,
                    'precision': precision_score,
                    'recall': recall_score,
                    'f1-score': f1_score,
                    'jaccard score': jaccard_score}

