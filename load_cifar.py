########################################################################
#
# Functions for downloading the CIFAR-10 data-set from the internet
# and loading it into memory.
#
# Implemented in Python 3.5
#
# Usage:
# 1) Set the variable data_path with the desired storage path.
# 2) Call maybe_download_and_extract() to download the data-set
#    if it is not already located in the given data_path.
# 3) Call load_class_names() to get an array of the class-names.
# 4) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels
#    for the training-set and test-set.
# 5) Use the returned data in your own program.
#
# Format:
# The images for the training- and test-sets are returned as 4-dim numpy
# arrays each with the shape: [image_number, height, width, channel]
# where the individual pixels are floats between 0.0 and 1.0.
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import tensorflow as tf
import numpy as np
from numpy import array
import pickle
import os
import download
from dataset import one_hot_encoded
import random
import matplotlib.pyplot as plt

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "data/CIFAR-10/"

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
_num_files_train = 5

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file

########################################################################
# Private functions for downloading, unpacking and loading data-files.


def _get_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set.
    If filename=="" then return the directory of the files.
    """

    return os.path.join(data_path, "cifar-10-batches-py/", filename)



def _unpickle(filename):
    """
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended the filename.
    """

    # Create full path for the file.
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='bytes')

    return data


def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.


def maybe_download_and_extract():
    """
    Download and extract the CIFAR-10 data-set if it doesn't already exist
    in data_path (set this variable first to the desired path).
    """

    download.maybe_download_and_extract(url=data_url, download_dir=data_path)


def load_class_names():
    """
    Load the names for the classes in the CIFAR-10 data-set.
    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    """

    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names


def load_training_data():
    """
    Load all the training-data for the CIFAR-10 data-set.
    The data-set is split into 5 data-files which are merged here.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_test_data():
    """
    Load all the test-data for the CIFAR-10 data-set.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    images, cls = _load_data(filename="test_batch")

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

########################################################################
train_size = 200
#val_size = 7500 #12.5% of all images
val_size = 40

#training and validation
maybe_download_and_extract()
class_names = load_class_names()
images_train, cls_train, labels_train = load_training_data()

#validation
images_val = images_train[:val_size,:,:,:]
cls_val = cls_train[:val_size]
labels_val = labels_train[:val_size,:]

#train
images_train = images_train[val_size:val_size+train_size,:,:,:]
cls_train = cls_train[val_size:val_size+train_size]
labels_train = labels_train[val_size:val_size+train_size,:]

'''
#train
images_train = images_train[val_size:,:,:,:]
cls_train = cls_train[val_size:]
labels_train = labels_train[val_size:,:]
'''
#test
images_test, cls_test, labels_test = load_test_data()

#shuffle the data
def randomize(dataset, cls, labels):
  permutation = np.random.permutation(cls.shape[0])
  shuffled_dataset = dataset[permutation,:,:,:]
  shuffled_cls = cls[permutation]
  shuffled_labels = labels[permutation,:]
  return shuffled_dataset, shuffled_cls, shuffled_labels
images_train, cls_train, labels_train = randomize(images_train, cls_train, labels_train)
images_test, cls_test, labels_test = randomize(images_test, cls_test, labels_test)
images_val, cls_val, labels_val = randomize(images_val, cls_val, labels_val)

'''imgplot = plt.imshow(images_val[215,:,:,:])
plt.show()'''

#augmentation
images_train_before_aug = images_train
def augment_images(images_train,aug_num):
    # train data augmentation - flip, pad and crop
    images_after_aug = []

    for im in images_train[0:aug_num]:
        #flip
        flip_im = np.random.randint(0, 2)
        if flip_im > 0:
            im = np.fliplr(im)
        #pad
        pad_num = 4
        padding = ((pad_num, pad_num), (pad_num, pad_num), (0, 0))
        im = np.pad(im, pad_width=padding, mode='reflect')
        #crop
        dx = 32
        dy = 32
        w = im.shape[0]
        h = im.shape[1]
        x = y = 0
        x = random.randint(0, w - dx)
        y = random.randint(0, h - dy)
        im_aug = im[x: x + dx, y: y + dy]

        images_after_aug.append(im_aug)

    return images_after_aug

#aug_num = images_train.shape[0]
aug_num = 50000 - val_size
images_train_after_aug = augment_images(images_train, aug_num)
images_train = array(images_train_after_aug)
images_train = np.concatenate((images_train_before_aug,images_train_after_aug), axis=0)
labels_train = np.concatenate((labels_train,labels_train[0:aug_num]), axis=0)

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Validation-set:\t\t{}".format(len(images_val)))
print("- Test-set:\t\t{}".format(len(images_test)))