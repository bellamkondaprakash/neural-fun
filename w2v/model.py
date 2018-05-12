import numpy as np
import tensorflow as tf

class SkipGramModel:
    def __init__(self, vocabulary_size, embedding_dimension):
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.train_data = tf.placeholder(tf.int32)
            self.valid_data = tf.placeholder(tf.int32)
            self.train_labels = tf.placeholder(tf.int32)
            self.valid_labels = tf.placeholder(tf.int32)
            
            self.embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_dimension],
                                  -1.0, 1.0))
            self.weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, self.embedding_dimension],
                                    stddev=1.0 / np.squart(self.embedding_dimension)))
            self.biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            self.embed_input = tf.nn.embedding_lookup(self.embeddings, self.train_data)

            self.loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights=self.weights, biases=self.biases,
                                           inputs=self.embed_input, labels=self.train_labels,
                                           num_sampled = self.vocabulary_size // 100,
                                           num_classes = self.vocabulary_size)

    def train(self, data)
