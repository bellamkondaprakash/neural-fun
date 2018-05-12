from collections import Counter

def data2ids(data, maxwords=100000, unknown_token=None):
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
    for elem, _ in counts:
        dictionary[elem] = len(dictionary)+1
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    transformed_data = []
    for elem in data:
        transformed_data.append(dictionary.get(elem, 0))

    return transformed_data, dictionary, reverse_dictionary

    
