from __future__ import division

import keras
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers
from keras import backend as K

import numpy as np

class Network():
    """ MLP that will be trained with pre-trained weights from
    the restricted Boltzmann Machine
    """
    def __init__(self):

        self.batch_size = 300
        self.classes = 10
        self.epochs = 30

        (self.x_train, self.y_train), \
            (self.x_test, self.y_test) = mnist.load_data()

        self.x_train_lab = self.y_train

        self.hidden_size = 32

    def images_to_vectors(self):
        """ We will normalize all values between 0 and 1 and we will
        flatten the 28x28 images into vectors of size 784. """
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255
        self.x_train = self.x_train.reshape((len(self.x_train),
                                             np.prod(self.x_train.shape[1:])))
        self.x_test = self.x_test.reshape((len(self.x_test),
                                           np.prod(self.x_test.shape[1:])))

        self.y_train = keras.utils.np_utils.to_categorical(self.y_train,
                                                           self.classes)
        self.y_test = keras.utils.np_utils.to_categorical(self.y_test,
                                                          self.classes)

    def build_model(self, weights):
        """ Defines and compiles the model.
        """
        self.model = Sequential()
        self.model.add(Dense(self.hidden_size,
                             input_shape=(len(self.x_train[0]), ),
                             weights=weights,
                             activation='sigmoid'))
        self.model.add(Dense(self.classes,
                             input_dim=self.hidden_size,
                             init='normal',
                             activation='softmax'))
        # sgd = keras.optimizers.SGD(lr=self.learning_rate)
        self.model.compile(loss='binary_crossentropy', optimizer='sgd',
                           metrics=['accuracy'])

    def train_model(self):
        fit = self.model.fit(self.x_train, self.y_train,
                             batch_size=self.batch_size,
                             nb_epoch=self.epochs,
                             verbose=1,
                             validation_data=(self.x_test, self.y_test))
        return fit

    def evaluate_model(self):
        performance = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print "Loss: ", performance[0]
        print "Accuracy: ", performance[1]
        return performance

