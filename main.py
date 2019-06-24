import matplotlib.pyplot as plt
from data_provider import dataProvider
from cross_validation import crossValidator
import numpy as np

from model import Keras_MLP
from model import MLP_ensemble

def main():
    dp = dataProvider()
    dp.test_for_nans()
    party_dict = dp.get_vote_dict()
    x_train, y_train = dp.get_train_xy(onehot_y=True)
    x_val, y_val = dp.get_val_xy(onehot_y=True)
    x_test = dp.get_test_xy()

    #model = Keras_MLP(drop_p=0.1)
    #model.fit(x_train,y_train,graphic=True)

    # Cross validation
    #cv = crossValidator(train_x=x_train, train_y=y_train,
    #                    num_of_folds=4, max_epochs=500)
    #cv.tune_dropout()
    #cv.tune_leaky_slope()
    #cv.tune_hidden_layers()
    #cv.rand_tune(iter=1000)

    ensamble = MLP_ensemble('saved_models', party_dict)
    #ensamble.score(x_val, y_val)
    #ensamble.predict_vote_division(x_test)


if __name__ == "__main__":
    main()