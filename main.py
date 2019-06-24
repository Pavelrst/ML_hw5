import matplotlib.pyplot as plt
from data_provider import dataProvider
from cross_validation import crossValidator
import numpy as np
import pandas as pd

from model import Keras_MLP
from model import MLP_ensemble

def main():
    dp = dataProvider()
    dp.test_for_nans()
    party_dict = dp.get_vote_dict()
    x_train, y_train = dp.get_train_xy(onehot_y=True)
    x_val, y_val = dp.get_val_xy(onehot_y=True)
    x_test = dp.get_test_xy()
    id_test = dp.get_test_id()

    # Cross validation Random search
    # cv = crossValidator(train_x=x_train, train_y=y_train, num_of_folds=4, max_epochs=500)
    # cv.rand_tune(iter=1000)

    # ensamble = MLP_ensemble('saved_models', party_dict)
    #ensamble.score(x_val, y_val)
    #ensamble.predict_winner(x_test)
    #ensamble.predict_vote_division(x_test)
    #ensamble.write_pred_to_csv(x_test, id_test)

    # Sanity check
    similarity_to_ido()

def similarity_to_ido():
    ido = pd.read_csv('ido_predictions.csv')
    pavel = pd.read_csv('test_predictions.csv')
    ido_party = ido.pop('Vote').values
    pavel_party = pavel.pop('Vote').values

    acc = 0
    for i,p in zip(ido_party, pavel_party):
        if i == p:
            acc += 1
    acc = acc/100
    print("Similarity to ido: ", acc, "%")

if __name__ == "__main__":
    main()