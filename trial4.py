'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
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

save_dir = os.path.join('models', 'saved_models')

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
model_name = 'mnemonic_model4_2.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

img_rows = 32
img_cols = 32
channels = 3
img_shape = (img_rows, img_cols, channels)

img = Input(shape=img_shape)

latent_seed_dim = 512

latent_augment_dim = latent_seed_dim + 128
# latent_augment_dim = latent_seed_dim + 10


# FIRST PORTION OF CNN
cifar_10_1 = Sequential()
cifar_10_1.add(Conv2D(32, (3, 3), padding='same',
                      input_shape=x_train.shape[1:]))
cifar_10_1.add(Activation('relu'))
cifar_10_1.add(Conv2D(32, (3, 3)))
cifar_10_1.add(Activation('relu'))
cifar_10_1.add(MaxPooling2D(pool_size=(2, 2)))
cifar_10_1.add(Dropout(0.25))

cifar_10_1.add(Conv2D(64, (3, 3), padding='same'))
cifar_10_1.add(Activation('relu'))
cifar_10_1.add(Conv2D(64, (3, 3)))
cifar_10_1.add(Activation('relu'))
cifar_10_1.add(MaxPooling2D(pool_size=(2, 2)))
cifar_10_1.add(Dropout(0.25))
cifar_10_1.add(Flatten())

temporary = cifar_10_1(img)

left_split = Sequential()
left_split.add(Dense(512))
left_split.add(Activation('relu'))

right_split = Sequential()
right_split.add(Dense(latent_seed_dim))
right_split.add(Activation('relu'))
ight_split.add(BatchNormalization(momentum=80))

seed = right_split(temporary)
latent = left_split(temporary)

# GENERATOR
tmp = os.path.join(save_dir, 'generator.h5')
generator_model = load_model(tmp)
generator_model.trainable = False

# LATENT REPRESENTATION FROM CLASSIFIER
tmp = os.path.join(save_dir, 'latent2.h5')
latent_model = load_model(tmp)
latent_model.trainable = False

# MNEMONIC DEVICE
mnist_img = generator_model(seed)
augment = latent_model(mnist_img)
# mnemonic_model = Model(inputs=latent_seed,
#                        outputs=augment)

# augmenter_model = Model(inputs=img,
#                         output=augment)

concat = Concatenate(-1)([latent, seed, augment])

cifar_10_2 = Sequential()
cifar_10_2.add(Dropout(0.5))
cifar_10_2.add(Dense(num_classes))
cifar_10_2.add(Activation('softmax'))

output = cifar_10_2(concat)

final_model = Model(inputs=img,
                    outputs=output)


# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
final_model.compile(loss='categorical_crossentropy',
                   optimizer=opt,
                   metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

final_model.fit(x_train, y_train,
               batch_size=batch_size,
               epochs=100,
               validation_data=(x_test, y_test),
               shuffle=True)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
final_model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = final_model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
