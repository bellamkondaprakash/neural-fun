import numpy as np
import tensorflow as tf

class W2VModel:
    def __init__(self, vocabulary_size, embedding_dimension, save_path=None):
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.graph = tf.Graph()
        self.initialized = False

        with self.graph.as_default():
            self.train_data = tf.placeholder(tf.int32)
            self.train_labels = tf.placeholder(tf.int32)
            
            self.embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_dimension],
                                  -1.0, 1.0))
            self.weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, self.embedding_dimension],
                                    stddev=1.0 / np.sqrt(self.embedding_dimension)))
            self.biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            self.embed_input = tf.nn.embedding_lookup(self.embeddings, self.train_data)

            self.loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights=self.weights, biases=self.biases,
                                           inputs=self.embed_input, labels=self.train_labels,
                                           num_sampled = self.vocabulary_size // 200,
                                           num_classes = self.vocabulary_size))
            self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)

            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keepdims=True))
            self.normalized_embeddings = self.embeddings / norm

            self.save_path = save_path
            if self.save_path is not None:
                self.saver = tf.train.Saver()

            

    def train(self, generator, steps, verbose=True):
        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            if self.save_path is not None:
                try:
                    self.saver.restore(session, self.save_path)
                    print("Weights and embeddings loaded from {}".format(self.save_path))
                except:
                    print("Saved model not found. Starting training from scratch.")
                
            average_loss = 0
            print("Starting training with {} steps.".format(steps))
            for step in range(1, steps):
                batch_data, batch_labels = next(generator)
                batch_data = np.asarray(batch_data)
                batch_labels = np.asarray(batch_labels).reshape((len(batch_labels), 1))
                _, loss = session.run([self.optimizer, self.loss],
                                      feed_dict={self.train_data: batch_data,
                                                 self.train_labels: batch_labels})
                average_loss += loss

                if verbose:
                    if step % 2000 == 0:
                        print("Step: {}: loss = {}".format(step, average_loss/2000))
                        average_loss = 0

            self.final_embeddings = self.normalized_embeddings.eval()
            self.saver.save(session, self.save_path)
