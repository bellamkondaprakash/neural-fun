This code implements skip-gram algorithm (and could be quite easily generalized for other algorithms)
for finding vector embeddings for words using TensorFlow. The project started from an exercise on
Udacity's Deep Learning course by Google and especially the algorithm itself is quite similar to
the one in the exercise. However, the batch generator part is greatly enhanced and the code is also
otherwise generalized to allow easy experimentation with different datasets.

An example of using the code can be found from `text8-embeddings.ipynb` Jupyter notebook where I use
the code to calculate vector embeddings from the [Text8](http://mattmahoney.net/dc/textdata) dataset.

The code has been tested on Python 3.5.3 and TensorFlow 1.6.0. It will probably also work with
other relatively recent versions of those libraries.

The data and pretrained weights for the applications are not included because of their large size.

I'm planning to add another example soon(ish): vector embeddings for recipe ingredients using a
slightly modified skip-gram algorithm. This can hopefully be used to find replacements for missing
ingredients in recipes.

TODO in the future:
* Add the recipe ingredient embeddings code.
* Generalize the batch generator for other algorithms (like Bag-of-Words).