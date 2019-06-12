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
        self.train_set = pd.read_csv(input_path+delimiter+'train_transformed.csv')
        self.val_set = pd.read_csv(input_path+delimiter+'validation_transformed.csv')
        self.test_set = pd.read_csv(input_path+delimiter+'test_transformed.csv')

        self.original_train = pd.read_csv(input_path + delimiter + 'train_backup.csv')

        # preparing dataset to model:
        self.y_train = self.train_set.pop('Vote').values
        self.y_val = self.val_set.pop('Vote').values
        self.y_test = self.test_set.pop('Vote').values

        self.x_train = self.train_set.values
        self.x_val = self.val_set.values
        self.x_test = self.test_set.values

        self.test_set_indices = self.test_set.index.values
        self.feature_names = self.train_set.columns


        self.vote_categories = None
        self.vote_numbers = None
        self.vote_dictionary = None

    def test_for_nans(self):
        assert sum([s.isna().sum().sum() for s in (self.train_set, self.val_set, self.test_set)]) == 0

    def get_vote_dict(self):
        '''
        :return: dictionary which maps 'Vote' category to numbers.
        '''
        party_names = self.original_train['Vote'].values
        party_nums = self.y_train

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
        return self.vote_dictionary

    def get_sets_as_pd(self):
        return self.train_set, self.val_set, self.test_set

    def get_train_xy(self, onehot_y=False):
        if not onehot_y:
            return self.train_set['Unnamed: 0'].values, self.x_train[:,1:], self.y_train
        else:
            dummy_y = np_utils.to_categorical(self.y_train)
            return self.train_set['Unnamed: 0'].values, self.x_train[:, 1:], dummy_y

    def get_val_xy(self, onehot_y=False):
        if not onehot_y:
            return self.val_set['Unnamed: 0'].values, self.x_val[:,1:], self.y_val
        else:
            dummy_y = np_utils.to_categorical(self.y_val)
            return self.val_set['Unnamed: 0'].values, self.x_val[:, 1:], dummy_y

    def get_test_xy(self, onehot_y=False):
        if not onehot_y:
            return self.test_set['Unnamed: 0'].values, self.x_test[:,1:], self.y_test
        else:
            dummy_y = np_utils.to_categorical(self.y_test)
            return self.test_set['Unnamed: 0'].values, self.x_test[:, 1:], dummy_y

    def get_feature_names(self):
        return self.test_set.columns[1:]
