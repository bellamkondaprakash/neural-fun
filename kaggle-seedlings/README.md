This is my code for the Kaggle plant seedlings classification challenge,
see https://www.kaggle.com/c/plant-seedlings-classification . I get
0.987 accuracy score on the validation dataset by averaging predictions from
three separate networks. Curiously, one of the three networks performs
slightly better on its own (customized Xception with 0.988 accuracy).

The part of the code that controls the training of the neural networks is in the train_on_seedlings.py file. Run it using
`python3 train_on_seedlings.py`. The cod that evaluates all the three models and also calculates average predictions over
the ensemble of all three models is in `ensemble-predict.py`. If run, it evaluates both training and validation accuracy
from the models and prints the results.

My approach is to use Keras and re-train several pre-defined networks from Keras applications. There is more code than would
be strictly needed to get the results, because my aim was to write code that can be easily reused for other image classification
tasks. If someone wants to try my approach on a different dataset, is is as simple as placing the training data in the data/ folder,
changing the number of categories in the train_on_seedlings.py file and re-training the nets.

The code has been tested on python 3.6.3, Keras 2.1.4 and TensorFlow 1.4.1. It will probably also run on other (relatively modern)
versions of the libraries.

Unfortunately the training dataset is not included in this repository. You must download it yourself from the Kaggle page and
place it in the data/ folder. splitvalid.py file can be used to then split the data into training and validation data. For this
task, I chose not to have separate testing data because the seedlings dataset was already pretty small (only 4750 images).
