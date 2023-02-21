import math
import numpy as np
import random
from utils.basic import *
from utils.local_io import *

# Assume the data is under directory ./data/mat, which includes *.mat files to store training data and a 
# split.info for train, val, and test.

# A split_info.csv file may look like:
# train,case_020
# val,case_020
# test,case_020

# A *.mat file should include all the training info.

class Dataset:
    def __init__(self, data_root, DEBUG=False):
        self.mat_dir = data_root
        self.DEBUG = DEBUG
        name_list = self._get_name_list()
        train_list = self._load_data(name_list['train'])
        self.train_sampler = Sampler(train_list)
        self.val_sampler = Sampler(self._load_data(name_list['val']))
        self.avg_sample_points = self._get_avg_sample_points(train_list)
        # if in test_mode, load test_data
        self.test_sampler = Sampler(self._load_data(name_list['test']))

    # return the file names for train, val, and test
    def _get_name_list(self, train_ratio=0.6, val_ratio=0.2):
        split_file = join_path(self.mat_dir, 'split.csv')
        # if there is a split.csv file exist, load the split file
        if os.path.exists(split_file):
            name_list = load_csv(split_file)
            train_list = name_list['train']
            val_list = name_list['val']
            test_list = name_list['test']

        # else split the data_new into train, validation and test, save it into data_dir/split.csv
        else:
            case_list = [case_name.split('.')[0] for case_name in listdir(self.mat_dir, postfix='.mat')]
            total_n = len(case_list)
            train_n = int(total_n * train_ratio)
            val_n = int(total_n * val_ratio)
            assert train_n + val_n < total_n
            test_n = total_n - train_n - val_n
            random.shuffle(case_list)

            train_list = case_list[:train_n]
            val_list = case_list[train_n:train_n + val_n]
            test_list = case_list[-test_n:]

            save_csv(split_file, [['train'] + train_list, ['val'] + val_list, ['test'] + test_list])

        return {'train': train_list, 'val': val_list, 'test': test_list}

    def _load_data(self, name_list, RANDOM=True):
        data_list = []
        count = 0
        for case_name in name_list:
            count += 1
            # load tmp_2d image and expand dimension
            mat_data = read_mat(join_path(self.mat_dir, case_name+'.mat'))
            data_list.append(self._reformat_mat(mat_data))
            if self.DEBUG and count >=3:
                break
        if RANDOM:
            random.shuffle(data_list)
        return data_list

    def _get_avg_sample_points(self, train_data_list):
        sample_points = np.zeros([576, 2])
        data_n = len(train_data_list)
        for data in train_data_list:
            sample_points += data['PriorShape']
        return sample_points/data_n

    @staticmethod
    def _reformat_mat(mat_data):
        keys = mat_data.keys()
        formatted_mat = {}
        for key in keys:
            key_data = mat_data[key]
            if key == 'Ideal_PX' or key == 'Panorama':
                # [1, 160, 576]
                key_data = np.expand_dims(key_data, axis=0)
            if key == 'Transfer_PX':
                # [1, 160, 576]
                key_data = np.flip(key_data, axis=0)
                key_data = np.expand_dims(key_data, axis=0)
            elif key == 'CBCT':
                # [256, 160, 288]
                key_data = (key_data - 2000) / 2000
                key_data = np.transpose(key_data, [1, 2, 0])
            elif key == 'Bone':
                # [256, 160, 288]
                key_data = np.transpose(key_data, [1, 2, 0])
            elif key == 'PriorShape':
                # [576, 2]
                key_data = (key_data[0:-1, :] + key_data[1:, :]) / 2
            elif key == 'MPR':
                # [batch_size, 80, 160, 576]
                key_data = (key_data - 2000) / 2000
                key_data = np.flip(key_data, axis=0)
            else:
                pass
            formatted_mat.update({key: key_data})
        return formatted_mat


class Sampler:
    def __init__(self, data_list):
        self.keys = data_list[0].keys()
        self.data_n = len(data_list)
        self.data_list = data_list
        self.batch_n = self.data_n
        self.counter = 0

    def get_batch(self, keys=None, id=None):
        keys = keys if keys else self.keys
        batch = {key: [] for key in keys}

        if id:
            key_data = self.data_list[id]
        else:
            key_data = self.data_list[self.counter]
            self.counter += 1
            if self.counter >= self.batch_n:
                self.counter = 0
        for key in self.keys:
            batch[key].append(key_data[key])
        # update counter
        return batch