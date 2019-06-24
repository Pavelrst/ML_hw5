import os
import pandas as pd
from sklearn.model_selection import train_test_split
import inspect
import re
from keras.utils import np_utils

class dataProvider():
    def __init__(self, input_path=''):
        self.train_size = None
        self.val_size = None
        self.test_size = None

        if input_path == '':
            delimiter = ''
        else:
            delimiter = '\\'
        self.train_set = pd.concat([pd.read_csv(input_path+delimiter+'train_transformed.csv'),
                                    pd.read_csv(input_path + delimiter + 'test_transformed.csv')])
        self.val_set = pd.read_csv(input_path+delimiter+'validation_transformed.csv')
        self.test_set = pd.read_csv(input_path + delimiter + 'unlabeled_set.csv')

        # prepare dict:
        party_names = self.train_set['Party'].values
        party_nums = self.train_set['Vote'].values

        num_list = []
        name_list = []
        for num, name in zip(party_nums, party_names):
            if num not in num_list:
                num_list.append(num)
                name_list.append(name)
            else:
                idx = num_list.index(num)
                assert name_list[idx] == name

        self.vote_dictionary = dict(zip(num_list, name_list))

        # get rid of parties names:
        self.train_set.pop('Party')
        self.val_set.pop('Party')

        # preparing dataset to model:
        self.y_train = self.train_set.pop('Vote').values
        self.y_val = self.val_set.pop('Vote').values

        self.x_train = self.train_set.values
        self.x_val = self.val_set.values
        self.x_test = self.test_set.values

        #self.test_set_indices = self.test_set.index.values
        self.feature_names = self.train_set.columns

    def test_for_nans(self):
        assert sum([s.isna().sum().sum() for s in (self.train_set, self.val_set, self.test_set)]) == 0

    def get_vote_dict(self):
        '''
        :return: dictionary which maps 'Vote' category to numbers.
        '''
        return self.vote_dictionary

    def get_sets_as_pd(self):
        return self.train_set, self.val_set, self.test_set

    def get_train_xy(self, onehot_y=False):
        if not onehot_y:
            return self.x_train, self.y_train
        else:
            dummy_y = np_utils.to_categorical(self.y_train)
            return self.x_train, dummy_y

    def get_val_xy(self, onehot_y=False):
        if not onehot_y:
            return self.x_val, self.y_val
        else:
            dummy_y = np_utils.to_categorical(self.y_val)
            return self.x_val, dummy_y

    def get_test_xy(self):
        return self.x_test


    def get_feature_names(self):
        return self.test_set.columns[1:]
