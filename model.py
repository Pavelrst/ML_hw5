from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import LeakyReLU, ReLU, Activation
from keras.layers import Dropout
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
import keras.optimizers
import matplotlib.pyplot as plt

class Keras_MLP():
    def __init__(self, n_hidden_list, num_features=9, lrelu_alpha=0.1,
                 drop_p=0.2, max_epochs=1000, activation='lrelu', scheduling=False, b_norm=False):
        # model params:
        self.scheduling = scheduling
        self.opt_patience = 5
        self.val_split = 0.05 # it used only for monitoring
        self.max_epochs = max_epochs
        # Create model
        self.model = Sequential()

        for idx, n in enumerate(n_hidden_list):
            if idx == 0:
                self.model.add(Dense(n, activation='linear', input_shape=(num_features,)))

                if b_norm:
                    self.model.add(BatchNormalization())

                if activation == 'lrelu':
                    self.model.add(LeakyReLU(alpha=lrelu_alpha))
                elif activation == 'tanh':
                    self.model.add(Activation('tanh'))
                else:
                    self.model.add(ReLU())
                self.model.add(Dropout(drop_p))
            else:
                self.model.add(Dense(n, activation='linear'))

                if b_norm:
                    self.model.add(BatchNormalization())

                if activation == 'lrelu':
                    self.model.add(LeakyReLU(alpha=lrelu_alpha))
                elif activation == 'tanh':
                    self.model.add(Activation('tanh'))
                else:
                    self.model.add(ReLU())
                self.model.add(Dropout(drop_p))

        self.model.add(Dense(13, activation='softmax'))

        opt = keras.optimizers.Adam(beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=None,
                                    decay=0.0,
                                    amsgrad=False)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])

        if self.scheduling:
            self.scheduler = Scheduler()


    def fit(self, x_train, y_train, graphic=False):
        # set early stopping monitor so the model stops training when it won't improve anymore
        early_stopping_monitor = EarlyStopping(patience=self.opt_patience)

        if self.scheduling:
            lrate = LearningRateScheduler(Scheduler().schedule, verbose=1)
            # train model
            hist = self.model.fit(x_train, y_train, validation_split=self.val_split,
                              epochs=self.max_epochs, callbacks=[early_stopping_monitor, lrate])
        else:
            # train model
            hist = self.model.fit(x_train, y_train, validation_split=self.val_split,
                                  epochs=self.max_epochs, callbacks=[early_stopping_monitor])

        if graphic:
            for key in hist.history:
                data = hist.history[key]
                epochs = range(len(hist.history[key]))
                target = [0.95]*len(hist.history[key])
                plt.plot(epochs, data, label=key)
            plt.plot(epochs, target, label='target = 0.95')
            plt.legend()
            plt.xlabel('epochs')
            plt.show()

    def predict(self, x_test):
        return self.model.predict(x_test)


class Scheduler:
    def __init__(self, lr=0.01, epochs_list=[10,20,30], decay=0.1):
        self.epochs_list = epochs_list
        self.decay = decay
        self.lr = lr
        self.current_milestone_idx = 0

    def schedule(self, epoch):
        if epoch == self.epochs_list[self.current_milestone_idx]:
            if self.current_milestone_idx + 1 < len(self.epochs_list):
                self.current_milestone_idx += 1
            self.lr = self.lr * self.decay
        return self.lr
