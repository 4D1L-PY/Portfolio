import pyLDAvis.gensim
import pyLDAvis.sklearn
from sklearn.decomposition import NMF, LatentDirichletAllocation
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
from src.nlp.text_preprocessor import np, plt, pd, gensim


class TopicModeling:

    def __init__(self, model_type, model_params, module_type='sklearn'):
        """

        :param model_type:
        :param model_params:
        :param module_type:
        """
        self.model_type = model_type
        self.model_params = model_params
        self.module_type = module_type
        self.model = None
        self.feature_names = None
        self.doc_topic_distr = None
        self.optimal_topic_number = None
        self.build_model()

    def build_model(self):
        """

        :return:
        """
        if self.module_type is 'sklearn':
            if self.model_type is 'lda':
                self.model = LatentDirichletAllocation(**self.model_params)
            elif self.model_type is 'nmf':
                self.model = NMF(**self.model_params)
        elif self.module_type is 'gensim':
            if self.model_type is 'lda':
                # Doc : https://radimrehurek.com/gensim/models/ldamodel.html
                self.model = gensim.models.LdaModel(**self.model_params)
                # lda_output = [t for t in lda.show_topics(num_topics=num_topics, formatted=True, log=False)]

    def fit(self, frequency_matrix):
        """

        :param frequency_matrix:
        :return:
        """
        self.model.fit(frequency_matrix)

    def plot_optimal_topic_number(self,
                                  xtrain,
                                  xtest,
                                  max_topics,
                                  metric='perplexity',
                                  step=5,
                                  tokens=None,
                                  vectorizer=None):
        """

        :param xtrain:
        :param xtest:
        :param max_topics:
        :param metric:
        :param step:
        :param tokens:
        :param vectorizer:
        :return:
        """
        model_params = {k: self.model_params[k] for k in self.model_params if k is not 'n_components'}
        topics = [n_topics for n_topics in range(step, max_topics, step)]
        scores = []
        for n_topics in topics:
            n_topics_model = LatentDirichletAllocation(n_components=n_topics, **model_params)
            n_topics_model.fit(xtrain)
            if metric is 'perplexity':
                n_topics_score = n_topics_model.perplexity(xtest)
            elif metric is 'coherence':
                if type(vectorizer) is list:
                    vectorizer_vocab = np.array([x for v in vectorizer for x in v.vocabulary_.keys()])
                else:
                    vectorizer_vocab = np.array([x for x in vectorizer.vocabulary_.keys()])
                n_topics_score = metric_coherence_gensim(measure='c_v',
                                                         top_n=n_topics,
                                                         topic_word_distrib=n_topics_model.components_,
                                                         dtm=xtest,
                                                         vocab=vectorizer_vocab,
                                                         texts=tokens.values,
                                                         return_mean=True)
            scores.append(n_topics_score)
        best_n_topics = topics[scores.index(min(scores))] if metric is 'perplexity' else topics[
            scores.index(max(scores))]
        self.optimal_topic_number = best_n_topics
        plt.xlabel('N topics')
        plt.ylabel(metric.capitalize())
        plt.plot(topics, scores, label='Best topic number = {}'.format(best_n_topics))
        plt.legend()
        plt.show()
        return

    def assign_topics(self, frequency_matrix, tokens_data):
        """

        :param frequency_matrix:
        :param tokens_data:
        :return:
        """
        self.doc_topic_distr = self.model.transform(frequency_matrix)
        topics_df = pd.DataFrame({'Tokens': tokens_data,
                                  'Topics': self.doc_topic_distr.argmax(axis=1) + 1})
        return topics_df

    def display_n_top_words_for_each_topics(self, n_top_words):
        """

        :param n_top_words:
        :return:
        """
        for topic_idx, topic in enumerate(self.model.components_):
            print(f"Topic {topic_idx}:")
            print(" ".join([self.feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

    def display_LDAvis(self, corpus, dictionary, n_top_words=10, enable_notebook=True):
        """

        :param corpus:
        :param dictionary:
        :param n_top_words:
        :param enable_notebook:
        :return:
        """
        if enable_notebook:
            pyLDAvis.enable_notebook()
        # Doc : https://pyldavis.readthedocs.io/en/latest/modules/API.html
        if self.module_type is 'sklearn':
            return pyLDAvis.sklearn.prepare(self.model, corpus, dictionary, R=n_top_words)  # mds='tsne'
        elif self.module_type is 'gensim':
            return pyLDAvis.gensim.prepare(self.model, corpus, dictionary, R=n_top_words)  # mds='tsne'
