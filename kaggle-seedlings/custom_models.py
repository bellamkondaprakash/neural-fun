import keras
from keras.preprocessing import image
from keras.layers import *
from keras.applications import inception_v3, xception, vgg19, vgg16, densenet, nasnet
import tools


def custom_inception(input_tensor, num_classes, weights=None, optimizer=keras.optimizers.SGD()):
    model_base = inception_v3.InceptionV3(include_top=False, input_tensor=input_tensor,
                                          weights='imagenet', pooling='avg')

    for ll in model_base.layers[1:]:
        ll.trainable = False
        ll.name += '_inc'

    inp_tensor = model_base.input
    x = model_base.output
    x = Dense(num_classes, activation='softmax', name='top_predictions_inc')(x)

    model = keras.Model(inputs=inp_tensor, outputs=x)
    try:
        model.load_weights(weights, by_name=False)
    except:
        print("Weight file {} could not be loaded. Using ImageNet weights.")

    model.preprocessor = inception_v3.preprocess_input

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def custom_xception(input_tensor, num_classes, weights=None, optimizer=keras.optimizers.SGD()):
    model_base = xception.Xception(include_top=False, input_tensor=input_tensor,
                                   weights='imagenet', pooling='avg')

    for ll in model_base.layers[1:]:
        ll.trainable = False
        ll.name += '_xc'

    inp_tensor = model_base.input
    x = model_base.output
    x = Dense(num_classes, activation='softmax', name='top_predictions_xc')(x)

    model = keras.Model(inputs=inp_tensor, outputs=x)
    try:
        model.load_weights(weights, by_name=False)
    except:
        print("Weight file {} could not be loaded. Using ImageNet weights.")

    model.preprocessor = xception.preprocess_input

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def custom_dn121(input_tensor, num_classes, weights=None, optimizer=keras.optimizers.SGD()):
    model_base = densenet.DenseNet121(include_top=False, input_tensor=input_tensor,
                                      weights='imagenet', pooling='avg')

    for ll in model_base.layers[1:]:
        ll.trainable = False
        ll.name += '_dn121'

    inp_tensor = model_base.input
    x = model_base.output
    x = Dense(num_classes, activation='softmax', name='top_predictions_dn121')(x)

    model = keras.Model(inputs=inp_tensor, outputs=x)
    try:
        model.load_weights(weights, by_name=False)
    except:
        print("Weight file {} could not be loaded. Using ImageNet weights.")

    model.preprocessor = densenet.preprocess_input

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
