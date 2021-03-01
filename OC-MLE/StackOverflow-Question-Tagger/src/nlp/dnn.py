from tensorflow import keras
import keras.backend as K
from src.nlp.text_preprocessor import np, plt, gensim


class DNN:

    def __init__(self, tokenizer=None, embedding_model=None, print_summary=True):
        # Keras tokenizer
        self.tokenizer = tokenizer
        # Embedding model
        self.embedding_model = embedding_model
        # Get embedding model parameters by embedding type (To do: add more models)
        if self.embedding_model is not None:
            if isinstance(self.embedding_model, gensim.models.word2vec.Word2Vec):
                # Trained word vectors
                self.trained_word_vectors = self.embedding_model.wv
                # Word2Vec vocabulary size
                self.vocab_size = len(self.trained_word_vectors.vocab.keys())
                # The size of the Word2Vec dense vector
                self.embedding_dim = self.embedding_model.vector_size
                # Max number of words in each complaint.
                self.max_sequence_length = self.embedding_model.vector_size
        # Keras main model
        self.model = keras.models.Sequential()
        self.print_summary = print_summary
        self.history = None

    def add_embedding_layer(self, **kwargs):
        """
        https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        """
        # Embedding matrix for the embedding layer
        embedding_matrix = np.zeros((self.vocab_size + 1, self.embedding_dim))
        # Fill embedding matrix with pre-trained Word2Vec
        for word, i in self.tokenizer.word_index.items():
            if word in self.trained_word_vectors:
                embedding_matrix[i] = self.trained_word_vectors[word]
        # Build embedding layer
        self.model.add(keras.layers.Embedding(input_dim=self.vocab_size + 1,
                                              output_dim=self.embedding_dim,
                                              weights=[embedding_matrix],
                                              input_length=self.max_sequence_length,
                                              trainable=False,
                                              **kwargs))

    def add_LSTM_layer(self, **kwargs):
        self.model.add(keras.layers.LSTM(**kwargs))

    def add_GRU_layer(self, **kwargs):
        self.model.add(keras.layers.GRU(**kwargs))

    def add_dense_layer(self, **kwargs):
        self.model.add(keras.layers.Dense(**kwargs))

    def add_dropout(self, dropout_rate, **kwargs):
        """
        https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
        """
        self.model.add(keras.layers.Dropout(dropout_rate, **kwargs))

    def compile_model(self, **kwargs):
        if self.print_summary:
            self.model.summary()
        self.model.compile(**kwargs)

    def fit(self, x_train, y_train, **kwargs):
        self.history = self.model.fit(x_train, y_train, **kwargs)

    def evaluate(self, x_test, y_test, **kwargs):
        results = self.model.evaluate(x_test, y_test, **kwargs)
        return results

    def plot_training_curve(self, value_type='accuracy'):
        """
        https://machinelearningmastery.com/diagnose-overfitting-underfitting-lstm-models/
        """
        plt.plot(self.history.history[value_type])
        plt.plot(self.history.history[f'val_{value_type}'])
        plt.title('Training curve')
        plt.ylabel(value_type.capitalize())
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


class Attention(keras.layers.Layer):

    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)
