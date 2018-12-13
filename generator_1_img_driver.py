from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import keras
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

from keras.models import load_model

import os

import matplotlib.pyplot as plt

save_dir = os.path.join('models', 'saved_models')

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# GENERATOR
tmp = os.path.join(save_dir, 'mnemonic_model1_3.h5')
generator_model = load_model(tmp)
generator_model.trainable = False

print(generator_model.layers)

# img_rows = 32
# img_cols = 32
# channels = 3
# img_shape = (img_rows, img_cols, channels)
#
# img = Input(shape=img_shape)
#
# # MNEMONIC DEVICE
#
# mnist_img = generator_model(img)
#
# gen = Model(img, mnist_img)
#
# # generate the images
# for i in range(10):
#     # sample 5 images for each class
#
#     sample = []
#
#     for ind, image in enumerate(x_train):
#         if y_train[ind] == i:
#             sample.append(ind)
#
#         if len(sample) == 10:
#             break
#
#     noise = x_train[sample]
#     mnist = gen.predict(noise)
#
#     r, c = 2, 10
#     fig, axs = plt.subplots(r, c)
#
#     # generate the images
#     for ind, index in enumerate(sample):
#         my_input = x_train[index]
#         my_output = mnist[ind]
#
#         my_output = 0.5 * my_output + 0.5
#
#
#         axs[0, ind].imshow((my_input/255.).astype('float32'))
#         axs[0, ind].axis('off')
#         axs[1, ind].imshow(my_output[:, :, 0], cmap='gray')
#         axs[1, ind].axis('off')
#
#     if not os.path.isdir("images4/"):
#         os.makedirs("images4/")
#     fig.savefig("images4/mnist_%d.png" % i)
#     plt.close()
