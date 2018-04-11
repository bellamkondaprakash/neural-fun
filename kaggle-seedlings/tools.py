"""Some tools to make the training of the models easy and streamlined.

"""

from numpy import *
import keras
from keras.preprocessing import image
import pickle


def extract_features(model, num_points, imsize,
                     generator, path,
                     num_categories, batch_size=32,
                     pickle_name=None, verbose=False):
    """Extracts features from a given model. Meant to be
    used without the top prediction layers of the network.

    Returns: a tuple (x,y) with len(x)=len(y)~num_points.
    Each element in x is an array of features extracted by
    propagating an image generatore by generator through the network.
    y is the corresponding label.
    """

    if pickle_name:
        try:
            with open(pickle_name, 'rb') as handle:
                x, y = pickle.load(handle)
            return (x, y)
        except:
            pass

    num_features = int(model.output.get_shape()[-1])
    x = array([]).reshape((0, num_features))
    y = array([]).reshape((0, num_categories))

    for batch in generator.flow_from_directory(path,
                                               target_size=(imsize, imsize),
                                               class_mode='categorical',
                                               batch_size=batch_size):
        pred = model.predict(batch[0], batch_size=batch_size)
        x = append(x, pred, axis=0)
        y = append(y, batch[1], axis=0)
        if verbose:
            print("Feature extraction progress: {} / {}".format(len(x), num_points))
        if len(x) >= num_points:
            break

    if pickle_name:
        with open(pickle_name, 'wb') as handle:
            pickle.dump((x, y), handle)

    return (x, y)


def get_generators(preprocess_fun, imsize=(224, 224), root_dir=None,
                   batch_size=12, rotation_range=30., flips=(True, True),
                   train_dir=None, valid_dir=None):
    """Creates two Keras ImageGenerators (train and valid) with
    parameters that are pre-chosen to be "convenient". The point is to
    get an output that can be fed directly to Keras fit_generator() function.

    Returns: a tuple (trainflow, validflow) with two iterators corresponding to
    image generators for the training data (with some image augmentations) and
    validation data (without any augmentations). Can be fed directly to Keras
    fit_generator() function.
    """

    try:
        if train_dir is not None:
            traindir = train_dir
        else:
            traindir = root_dir + '/train'
    except:
        print('Train directory invalid')

    try:
        if valid_dir is not None:
            validdir = valid_dir
        else:
            validdir = root_dir + '/valid1'
    except:
        print('Valid directory invalid')

    if train_dir is not None:
        traindir = train_dir

    traingen = image.ImageDataGenerator(rotation_range=rotation_range,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=0.05,
                                        fill_mode='reflect',
                                        horizontal_flip=flips[0],
                                        vertical_flip=flips[1],
                                        preprocessing_function=preprocess_fun)
    trainflow = traingen.flow_from_directory(traindir, target_size=imsize,
                                             batch_size=batch_size,
                                             class_mode='categorical',
                                             shuffle=True)

    validgen = image.ImageDataGenerator(preprocessing_function=preprocess_fun)
    validflow = validgen.flow_from_directory(validdir,
                                             target_size=imsize,
                                             batch_size=batch_size,
                                             class_mode='categorical',
                                             shuffle=True)

    return (trainflow, validflow)


def train_custom_model(model, train_dir, valid_dir,
                       protocol=[(10, False, None)],
                       imsize=(224, 224), weights=None,
                       steps_per_epoch=100, batchsize=8):
    """Trains model according to protocol argument with data from train_dir
    and valid_dir.

    protocol is a list of tuples with three parameters:
    (number of epochs, only_top, optimization algorithm).
    Each protocol is used for one training run and the weights
    are saved after every run weights is defined. At the beginning,
    weights from the weights argument are loaded into the model.
    """

    batchsize = batchsize
    steps_per_epoch = steps_per_epoch

    traing, validg = get_generators(model.preprocessor, imsize, batch_size=batchsize,
                                    rotation_range=30., flips=(True, True),
                                    train_dir=train_dir, valid_dir=valid_dir)

    try:
        model.load_weights(weights, by_name=False)
    except:
        print("Weight file {} could not be loaded. Using model's default weights.".format(weights))

    print("Training in {} phases...".format(len(protocol)))
    for i, phase in enumerate(protocol):
        epochs, only_top, optimizer = phase

        print("Doing phase {} with {} epochs, only_top={}".format(
            i, epochs, only_top))

        if optimizer is None:
            optimizer = keras.optimizers.Adadelta()

        for ll in model.layers[1:]:
            ll.trainable = not only_top
            if ll.name[:3] == 'top':
                ll.trainable = True

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit_generator(traing, steps_per_epoch=steps_per_epoch,
                            epochs=epochs, validation_data=validg, workers=4)

        if weights is not None:
            model.save_weights(weights)
