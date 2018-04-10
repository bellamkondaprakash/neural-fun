"""Split the data into train and validation sets.

Three directories are defined:
-sourcedir, where all of the data is located
-traindir, where training data will be symlinked from sourcedir
-vdir1, where training data will be symlinked from sourcedir

If directories are not initially empty, emptydirs variable decides if
they will be emptied or not. rvalid1 parameter is the valid-to-all data ratio.
Rest of the data will go in the train directory.

"""

import os
import random
from shutil import copy, rmtree
from contextlib import suppress
from numpy.random import randint


"""Parameters that define the behaviour of the split are here.
Adjust as needed
"""

emptydirs = True  # Should we empty the target directories if not empty
rvalid1 = 0.15  # valid-to-all data ratio

sourcedir = './data/alldata/'  # Directory with all the data
traindir = './data/train/'  # Directory for the train data
vdir1 = './data/valid1/'  # Directory for the validation data

verbose = True  # Print some information about the progress

"""Rest is the the code
"""

SEED = 1  # Seed for the random split, used in random.shuffle() below
random.seed(SEED)

categories = os.listdir(sourcedir)
nfiles = [len(os.listdir(sourcedir+cc)) for cc in categories]

if verbose:
    print("List of categories and number of files in them:")
    for name, num in zip(categories, nfiles):
        print("{}: {} images".format(name, num))
    print("Total: {} images".format(sum(nfiles)))

    print("Splitting with ratio {}...".format(rvalid1))

with suppress(FileExistsError):
    os.mkdir(traindir)
with suppress(FileExistsError):
    os.mkdir(vdir1)
for cc in categories:
    with suppress(FileExistsError):
        os.mkdir(traindir+cc)
    with suppress(FileExistsError):
        os.mkdir(vdir1+cc)

for cc in enumerate(categories):
    if os.listdir(traindir+cc[1]):
        if emptydirs:
            rmtree(traindir+cc[1])
            os.mkdir(traindir+cc[1])
        else:
            raise OSError(traindir+cc[1] + ' is not empty')
    if os.listdir(vdir1+cc[1]):
        if emptydirs:
            rmtree(vdir1+cc[1])
            os.mkdir(vdir1+cc[1])
        else:
            raise OSError(traindir+cc[1] + ' is not empty')

    files = os.listdir(sourcedir+cc[1])
    random.shuffle(files)
    for ff in files[:(int(rvalid1*nfiles[cc[0]]))]:
            os.symlink(os.path.abspath(sourcedir+cc[1]+'/'+ff),
                       os.path.abspath(vdir1+cc[1])+'/'+ff)
    for ff in files[(int(rvalid1*nfiles[cc[0]])):]:
            os.symlink(os.path.abspath(sourcedir+cc[1]+'/'+ff),
                       os.path.abspath(traindir+cc[1])+'/'+ff)
if verbose:
    print("...done")
