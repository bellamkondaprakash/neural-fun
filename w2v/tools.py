from collections import Counter
import numpy as np


class DataContainer:
    def __init__(self, data, maxwords=100000, unknown_token=None):
        self.data, self.dictionary, self.inverse_dictionary = words2ids(data,
                                                                        maxwords,
                                                                        unknown_token)
        self.embedding_matrix = None

    def add_embeddings(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix

    def distance_to(self, word):
        index = dictionary[word]
        return np.asarray([np.linalg.norm(self.embedding_matrix[index]-self.embedding_matrix[ind2])
                           for ind2 in range(len(self.embedding_matrix))])

    def closest_to(self, word):
        dists = distance_to(self.embedding_matrix, self.dictionary[word])
        return [self.inverse_dictionary[ind] for ind in np.argsort(dists)]

    def closest_to_vector(self, vector):
        dists = np.asarray([np.linalg.norm(vector-self.embedding_matrix[ind2])
                            for ind2 in range(len(self.embedding_matrix))])
        return np.argsort(dists)


def words2ids(data, maxwords=100000, unknown_token=None):
    """"Takes in a list of some elements and assigns each different element
    unique ID. Then returns data encoded as IDs and a dictionary and
    a reverse dictionary for translating between elements and their
    encodings.

    IDs are created so that the most common element has ID=1 and
    elements that are not assigned a unique ID get ID=0.
    """

    counts = Counter(data).most_common(maxwords)
    dictionary = dict()
    if unknown_token is not None:
        dictionary[unknown_token] = 0
    else:
        dictionary["_NOT_FOUND_"] = 0
    for elem, _ in counts:
        dictionary[elem] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    transformed_data = []
    for elem in data:
        transformed_data.append(dictionary.get(elem, 0))

    return transformed_data, dictionary, reverse_dictionary


def distance_to(embeddings, index):
    return np.asarray([np.linalg.norm(embeddings[index]-embeddings[ind2]) for ind2 in range(len(embeddings))])


def closest_to(embeddings, word, dictionary, inverse_dictionary):
    dists = distance_to(embeddings, dictionary[word])
    return [inverse_dictionary[ind] for ind in np.argsort(dists)]


def closest_to_vector(embeddings, vector):
    dists = np.asarray([np.linalg.norm(vector-embeddings[ind2])
                        for ind2 in range(len(embeddings))])
    return np.argsort(dists)
