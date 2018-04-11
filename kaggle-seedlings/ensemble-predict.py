"""Compute the accuracy of all the trained models on both training and validation data.
Also compute the accuracy of an averaged prediction from all models.
"""

import sys
import numpy as np
import keras
from keras.preprocessing import image
import custom_models as models
import tools

root_dir = './data'
train_dir = root_dir + '/train'
valid_dir = root_dir + '/valid1'

imsize = (224, 224)
num_categories = 12
batch_size = 32

input_layer = keras.layers.Input(shape=(*imsize, 3), name='input_layer')

nets = [models.custom_inception(input_layer, num_categories,
                                weights='weights/weights_inception_fc2.hdf5', fc=2),
        models.custom_xception(input_layer, num_categories,
                               weights='weights/weights_xception_fc2.hdf5', fc=2),
        models.custom_dn121(input_layer, num_categories,
                            weights='weights/weights_dn121_fc2.hdf5', fc=2)]


def predict_all(directory):
    preds = []
    for i, net in enumerate(nets):
        print("Predicting with net number {}".format(i))

        generator = image.ImageDataGenerator(preprocessing_function=net.preprocessor).flow_from_directory(
            directory, target_size=imsize, batch_size=batch_size, class_mode='categorical', shuffle=False)
        num_batches = generator.n // batch_size
        preds0 = np.zeros((generator.n, num_categories))
        y = np.zeros((generator.n, num_categories))

        sys.stdout.write("Predictions done: ")
        numdone = ""
        for ind, batch in enumerate(generator):

            sys.stdout.write("\b"*len(numdone))
            numdone = str(ind*batch_size)
            sys.stdout.write(numdone)
            sys.stdout.flush()

            newpred = net.predict(batch[0])
            y[(ind*batch_size):((ind+1)*batch_size)] = batch[1]
            preds0[(ind*batch_size):((ind+1)*batch_size)] = newpred
            if ind >= num_batches:
                break
        preds.append(preds0)
        sys.stdout.write("\n")
    return (preds, y)


preds_train, y_train = predict_all(train_dir)
preds_valid, y_valid = predict_all(valid_dir)

avg_train = sum(preds_train)/3
avg_valid = sum(preds_valid)/3

yind_train = [np.argmax(pp) for pp in y_train]
yind_valid = [np.argmax(pp) for pp in y_valid]

predind_train = [np.argmax(pp) for pp in avg_train]
predind_train_all = [[np.argmax(pp) for pp in preds0]
                     for preds0 in preds_train]
predind_valid = [np.argmax(pp) for pp in avg_train]
predind_valid_all = [[np.argmax(pp) for pp in preds0]
                     for preds0 in preds_valid]

corrects_train = [xx == yy for (xx, yy) in zip(predind_train, yind_train)]
corrects_train_all = [[xx == yy for (xx, yy) in zip(
    predindx0, yind_train)] for predindx0 in predind_train_all]
corrects_valid = [xx == yy for (xx, yy) in zip(predind_valid, yind_valid)]
corrects_valid_all = [[xx == yy for (xx, yy) in zip(
    predindx0, yind_valid)] for predindx0 in predind_valid_all]

accuracy_train = sum(corrects_train)/len(corrects_train)
accuracy_train_all = [sum(cc)/len(cc) for cc in corrects_train_all]
accuracy_valid = sum(corrects_valid)/len(corrects_valid)
accuracy_valid_all = [sum(cc)/len(cc) for cc in corrects_valid_all]

print("Accuracy of individual nets on training data:", accuracy_train_all)
print("Accuracy of averaged prediction on training data:", accuracy_train)

print("Accuracy of individual nets on validation data:", accuracy_valid_all)
print("Accuracy of averaged prediction on validation data:", accuracy_valid)
