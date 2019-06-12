from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.callbacks import LearningRateScheduler
import keras.optimizers

class Keras_MLP():
    def __init__(self, n_hidden_list=[150,100,50],num_features=9, lrelu_alpha=0.1, drop_p=0.2):
        # Create model
        self.model = Sequential()
        self.model.add(Dense(200, activation='linear', input_shape=(num_features,)))
        self.model.add(LeakyReLU(alpha=lrelu_alpha))
        self.model.add(Dropout(drop_p))

        for n in n_hidden_list:
            self.model.add(Dense(n, activation='linear'))
            self.model.add(LeakyReLU(alpha=lrelu_alpha))
            self.model.add(Dropout(drop_p))

        self.model.add(Dense(13, activation='softmax'))

        opt = keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        self.scheduler = Scheduler(lr=0.01, epochs_list=[10,20,30], decay=0.1)

    def fit(self, x_train, y_train):
        # set early stopping monitor so the model stops training when it won't improve anymore
        early_stopping_monitor = EarlyStopping(patience=3)

        lrate = LearningRateScheduler(Scheduler().schedule, verbose=1)
        # train model
        self.model.fit(x_train, y_train, validation_split=0.2, epochs=5, callbacks=[early_stopping_monitor, lrate])

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
