"""Train some models with the seedlings dataset.
Here we use the same training protocol for all models.
The training protocol is a result of trial and error,
it is also possible that a better one can be found.
I find that Adadelta optimizer does a good job making the
valid accuracy converge fast, but SGD is better for fine
tuning.

The batch_size is set to pretty low by default to allow
training on a GPU with only 4GB of memory.
"""

import keras
from keras.optimizers import *
import custom_models as models
import tools


root_dir = './data'
train_dir = root_dir + '/train'
valid_dir = root_dir + '/valid1'


train_inception = True
train_xception = True
train_densenet = True

imsize = (224, 224)
num_categories = 12
batch_size = 8

train_protocol = [(100, False, SGD(lr=0.01, momentum=0.5)),
                  (100, False, SGD(lr=0.001, momentum=0.5)),
                  (100, False, SGD(lr=0.0001, momentum=0.5)),
                  (50, True, SGD(lr=0.0001, momentum=0.5))]


input_layer = keras.layers.Input(shape=(*imsize, 3), name='input_layer')

if train_inception:
    model_inception = models.custom_inception(input_layer, num_categories,
                                              weights='weights/weights_inception_fc2.hdf5', fc=2)

    tools.train_custom_model(model_inception, train_dir, valid_dir,
                             protocol=train_protocol, imsize=imsize,
                             weights='weights/weights_inception_fc2.hdf5',
                             batchsize=batch_size)
    del model_inception
    
if train_xception:
    model_xception = models.custom_xception(input_layer, num_categories,
                                            weights='weights/weights_xception_fc2.hdf5', fc=2)

    tools.train_custom_model(model_xception, train_dir, valid_dir,
                             protocol=train_protocol, imsize=imsize,
                             weights='weights/weights_xception_fc2.hdf5',
                             batchsize=batch_size)
    del model_xception

if train_densenet:
    model_dn121 = models.custom_dn121(input_layer, num_categories,
                                      weights='weights/weights_dn121_fc2.hdf5', fc=2)

    tools.train_custom_model(model_dn121, train_dir, valid_dir,
                             protocol=train_protocol, imsize=imsize,
                             weights='weights/weights_dn121_fc2.hdf5',
                             batchsize=batch_size)
    del model_dn121
