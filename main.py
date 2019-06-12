import matplotlib.pyplot as plt
from data_provider import dataProvider
from cross_validation import crossValidator
import numpy as np

from model import Keras_MLP

def main():
    dp = dataProvider()
    dp.test_for_nans()
    id_train, x_train, y_train = dp.get_train_xy(onehot_y=True)
    id_val, x_val, y_val = dp.get_val_xy(onehot_y=True)
    id_test, x_test, y_test = dp.get_test_xy(onehot_y=True)

    assert set(id_train).intersection(set(id_val)) == set()
    assert set(id_val).intersection(set(id_test)) == set()
    assert set(id_test).intersection(set(id_train)) == set()

    #model = Keras_MLP()
    #model.fit(x_train,y_train)

    # Cross validation
    cv = crossValidator(train_x=x_train, train_y=y_train, num_of_folds=4)
    cv.tune_dropout()


if __name__ == "__main__":
    main()