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
    print("Cross validation Random search with train set of size: ", len(y_train), " val set size:", len(y_val))
    #cv = crossValidator(train_x=x_train, train_y=y_train, num_of_folds=4, max_epochs=500)
    #cv.rand_tune(iter=10000)

    ensamble = MLP_ensemble('saved_models', party_dict)
    ensamble.score(x_val, y_val)
    ensamble.predict_winner(x_test)
    ensamble.predict_vote_division(x_test)
    ensamble.write_pred_to_csv(x_test, id_test)

    # Sanity check
    # plot_random_models()
    similarity_to()

def plot_random_models():
    with open('random_models.txt') as f:
        lines = f.readlines()
        acc_list = []
        f1_list = []
        for line in lines:
            try:
                acc_list.append(float(line.split(':')[1].split(',')[0]))
                f1_list.append(float(line.split(':')[2].split(',')[0]))
            except:
                print("bad line")

        iter_list = range(len(acc_list))
        target = [0.945]*len(acc_list)
        plt.scatter(iter_list, acc_list, label='Accuracy', s=1)
        plt.scatter(iter_list, f1_list, label='F1 score', s=1)
        plt.plot(iter_list, target, label='target 0.945', c='r')
        plt.legend()
        plt.ylabel('Accuracy, F1 score')
        plt.xlabel('Iteration of random search')
        plt.ylim(0.5,1)
        plt.show()

def similarity_to():
    ido = pd.read_csv('chen_predictions.csv')
    pavel = pd.read_csv('vote_predictions.csv')
    ido_party = ido.pop('PredictVote').values
    pavel_party = pavel.pop('PredictVote').values

    acc = 0
    for i,p in zip(ido_party, pavel_party):
        if i == p:
            acc += 1
    acc = acc/100
    print("Similarity to chen: ", acc, "%")

if __name__ == "__main__":
    main()