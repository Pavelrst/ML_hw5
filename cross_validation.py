import numpy as np
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from model import Keras_MLP
from sklearn.metrics import f1_score

PLOTS_PATH = 'Cross_valid_plots'

class crossValidator():
    '''
    This class perfroms k-Fold Cross-Validation on training set to choose
    the proper hyperparameters for each model. It selects the models
    with best performance.
    '''
    def __init__(self, train_x, train_y, num_of_folds):
        self.set_x = train_x
        self.set_y = train_y
        self.k = num_of_folds
        self.kf = KFold(n_splits=num_of_folds)
        self.best_model = None
        self.num_of_classes = 13


    def tune_dropout(self, max_iter = 500, graphic=True):
        scores1 = []

        p_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        for p in p_list:
            print("training for p=", p)
            avg_score1 = 0

            mlp1 = Keras_MLP()

            for train_index, test_index in self.kf.split(self.set_x):
                x_train, x_test = self.set_x[train_index], self.set_x[test_index]
                y_train, y_test = self.set_y[train_index], self.set_y[test_index]
                mlp1.fit(x_train, y_train)

                y_pred = mlp1.predict(x_test)
                score1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='weighted')

                avg_score1 += score1

            scores1.append(avg_score1 / self.k)

        if graphic:
            plt.plot(p_list, scores1, label="1 hidden, relu")

            plt.title('hidden layers size tuning for MLP')
            plt.ylabel('Accuracy')
            plt.xlabel('hidden layers size')
            plt.legend()
            fig = plt.gcf()
            path = PLOTS_PATH + '\\' + 'mlp_h_fig.png'
            fig.savefig(path, bbox_inches='tight')
            plt.show()

