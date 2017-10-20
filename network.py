from keras.layers import Input, Dense, MaxPooling2D, Flatten, Dropout, Conv2D, BatchNormalization, Activation
from keras.optimizers import Adagrad
from keras.models import Model
from keras import regularizers
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.engine.topology import Layer
import numpy as np
from file_operation import read_h5file, write_h5file


class NeuralNetwork:
    """ Abstract class for all kinds of neural network

    Do not instantiate this class directly"""

    def __init__(self):
        """initialize new neural network with default arguments"""
        self.optimizer = None
        self.loss = None
        self.metrics = None
        self.monitor = None

        self.model = None
        self.network_type = None

        self.model_path = None
        self.log_path = None
        self.stats_path = None

        self.callbacks = None
        self.init = None

    def build(self):
        """set default attributes of model

        Model loss, optimizer, metrics, ordering, monitor
        Callback information:
            Early stopping, model_path, logger_path
        Training statistic information:
            stats_path

        if you want to use specified path for file please set the correct value for these variables before calling build
        """
        if not self.model_path:
            self.model_path = './model.h5'
        if not self.log_path:
            self.log_path = './training_log.csv'
        if not self.stats_path:
            self.stats_path = './training_statistic.h5'

        self.optimizer = 'adam'
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']

        # self.image_dim_ordering = K.image_dim_ordering()
        self.monitor = 'val_acc'

        self.init = 'glorot_uniform'

        early_stopping = EarlyStopping(monitor=self.monitor,
                                       patience=30,
                                       verbose=0,
                                       mode='auto')

        checkpoint = ModelCheckpoint(self.model_path, monitor=self.monitor)
        logger = CSVLogger(self.log_path)
        lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                                       patience=5, min_lr=1e-10)

        self.callbacks = [early_stopping, checkpoint, logger, lr_reducer]
        self.optimizer = Adagrad(lr=0.01, epsilon=1e-8, decay=0.0)

    def compile(self, **kwargs):
        """Compile model with given arguments otherwise using default value.

        Arguments(optional):
            optimizer, loss, metrics.
        """
        if "optimizer" in kwargs:
            self.optimizer = kwargs['optimizer']
        if 'loss' in kwargs:
            self.loss = kwargs["loss"]
        if 'metrics' in kwargs:
            self.metrics = kwargs['metrics']
        if 'init' in kwargs:
            self.init = kwargs['init']

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

    def set_callbacks(self, **kwargs):
        """Set the attributes of callback instead of using default value.

        Arguments(optional)
            monitor, model_path, log_path, stats_path.
        """
        if 'monitor' in kwargs:
            self.monitor = kwargs['monitor']
        if 'model_path' in kwargs:
            self.model_path = kwargs['model_path']
        if 'log_path' in kwargs:
            self.log_path = kwargs['log_path']
        if 'stats_path' in kwargs:
            self.stats_path = kwargs['stats_path']

    def train(self,
              x_train, y_train,
              batch_size=128,
              epoch=100,
              callbacks=None,
              **kwargs):
        """Training model with given data

        Assume the data format has been corrected to fit the backend

        Arguments:
            x_train: numpy array
                containing intensity information
            y_train: numpy array
                containing labels information(remember to do np_util.to_categorical operation)
            batch_size: int
                The number of samples being passed to model at a time
            epoch: int
                The maxmium number of cycle
            callbacks: list of Callback
                The callback function, leave it empty for using default value
            x_valid, y_valid(optional): two numpy array of data which can be used for validation.
                Leave it empty to use part of x_train and y_train as validation dataset.
        """
        print(self.model.summary())

        if not callbacks:
            callbacks = self.callbacks

        if ("x_valid" in kwargs) and ('y_valid' in kwargs):
            print("start training")
            x_valid = kwargs['x_valid']
            y_valid = kwargs['y_valid']

            self.model.fit(x_train, y_train,
                           batch_size=batch_size,
                           epochs=epoch,
                           verbose=1,
                           validation_data=(x_valid, y_valid),
                           callbacks=callbacks)
        else:
            self.model.fit(x_train, y_train,
                           batch_size=batch_size,
                           epochs=epoch,
                           verbose=1,
                           validation_split=0.1,
                           callbacks=callbacks)

        return self.model

    def evaluate_generator(self, x_test, y_test,
                           batch_size=1024, verbose=1):
        """ Return the accuracy of model given test data and its true labels.

        Arguments:
            x_test: numpy array
                intensity information of test data
            y_test: numpy array
                true label information of test data
            batch_size: int
                the number of samples are passed to model at a time
            verbose: 0 or 1
                show the process or not
        Return:
            score: float
                the accuracy of the model(compare the true labels with the predicted labels generated by model)
        """
        score = self.model.evaluate(x_test, y_test, batch_size=batch_size, verbose=verbose)
        return score

    def predict(self, x_test, batch_size=1024, verbose=1):
        """ Return the probability information given test data.

        Arguments:
            x_test: numpy array
                intensity information of test data
            batch_size: int
                the number of samples are passed to model at a time
            verbose: 0 or 1
                show the process or not

        Return:
            predicted probability: numpy array(2d: (None, nb_classes)
                each column represents a probability for the given row belonging to this label

        """
        return self.model.predict(x_test, batch_size, verbose)

    def predict_generator(self, x_generator, step):
        return self.model.predict_generator(x_generator, step, verbose=1)

    def predict_on_batch(self, x_test):
        """ Returns predictions for a single batch of samples..

        Arguments:
            x_test: numpy array
                intensity information of test data

        Return:
            predicted probability: numpy array(2d: (None, nb_classes)
                each column represents a probability for the given row belonging to this label
        """
        return self.model.predict_on_batch(x_test)

    def save_model(self, model_path=None):
        """Save the model into a HDF5.

        Arguments:
            model_path: string
                The path to store the model. Leave this empty to save to default path.
        """
        model_path = model_path if model_path else self.model_path
        self.model.save_weights(model_path)

    def load_model(self, model_path):
        """Load the model from a HDF5.

        Arguments:
            model_path: string
                the path of a HDF5 file where the model information is stored
        """
        print("Load model from {}".format(model_path))
        self.build()
        self.model.load_weights(model_path)

    def save_stats(self, stats_path=None):
        """Save network and training statistics.

        Arguments:
            stats_path: string
                The path to store the statistic information. Leave this empty to save to default path.
        """
        self.stats_path = stats_path if stats_path else self.stats_path

        # wrap general attributes
        self.attrs = {}
        self.attrs['network_type'] = self.network_type
        self.attrs['monitor'] = self.monitor
        self.attrs['loss'] = self.loss
        self.attrs['optimizer'] = type(self.optimizer).__name__
        write_h5file(stats_path, self.attrs)

    def load_stats(self, stats_path):
        """Load network and training statistics."""
        self.stats = read_h5file(stats_path)
        self.monitor = self.stats['monitor']
        self.network_type = self.stats['network_type']


class TriplanarCNN(NeuralNetwork):
    """Abstract class for Triplanar neural network"""

    def __init__(self, patch_size, nb_classes, nb_channels=3):
        super(TriplanarCNN, self).__init__()
        self.network_type = 'Triplanar'
        self.patch_size = patch_size
        self.nb_channels = nb_channels
        self.nb_classes = nb_classes

    def _bn_relu(self, x):
        x = BatchNormalization(axis=1)(x)
        # x = K.relu(x)
        x = Activation(K.relu)(x)
        return x

    def build(self):
        NeuralNetwork.build(self)

        regularizer = regularizers.l2(0.01)

        nb_filter = 32
        tf_format = 'channels_first'

        # input_shapes = (self.nb_channels, self.patch_size, self.patch_size)

        input = Input((self.nb_channels, self.patch_size, self.patch_size, ))

        x = Conv2D(nb_filter, 3, padding='valid', data_format=tf_format,
                   kernel_initializer=self.init, kernel_regularizer=regularizer)(input)
        x = self._bn_relu(x)

        x = Conv2D(nb_filter*2, 3, padding='valid', data_format=tf_format,
                   kernel_initializer=self.init, kernel_regularizer=regularizer)(x)
        x = self._bn_relu(x)
        x = MaxPooling2D(2, 2, data_format=tf_format)(x)

        x = Conv2D(nb_filter*2, 3, padding='valid', data_format=tf_format,
                   kernel_initializer=self.init, kernel_regularizer=regularizer)(x)
        x = self._bn_relu(x)

        x = Conv2D(nb_filter*4, 3, padding='valid', data_format=tf_format,
                   kernel_initializer=self.init, kernel_regularizer=regularizer)(x)
        x = self._bn_relu(x)

        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(256, kernel_initializer=self.init, kernel_regularizer=regularizer)(x)
        output = Dense(self.nb_classes, activation='softmax', 
                        kernel_initializer=self.init, kernel_regularizer=regularizer)(x)

        model = Model(inputs=input, outputs=output)

        self.model = model


if __name__ == '__main__':
    net = TriplanarCNN(28, 2, 3)
    net.build()
    net.model.summary()