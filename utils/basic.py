import os
import re
import numpy as np


# put path together
def join_path(path, *paths):
    return os.path.join(path, *paths)


# list files under the folder
def listdir(path, postfix=None, prefix=None):
    files = os.listdir(path)
    file_list_new = []
    for f in files:
        if postfix and not prefix:
            if f.endswith(postfix):
                file_list_new.append(f)
        elif prefix and not postfix:
            if f.startswith(prefix):
                file_list_new.append(f)
        elif prefix and postfix:
            if f.endswith(postfix) and f.startswith(prefix):
                file_list_new.append(f)
        else:
            if not f.startswith('.'):
                file_list_new.append(f)
    if len(file_list_new) == 0:
        raise ValueError('finding no file in ', path)
    return sorted(file_list_new)


# check directory whether exists:
# if yes, return the dir path; if not, make the dir return the dir path
def check_dir(dir_path, INFO=False):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        if INFO:
            print('Making new directory: %s' % dir_path)
    return dir_path


# get case id from case name
def get_id_by_name(name):
    id = re.findall("\d+", name)[0]
    id = int(id)
    return id


def print_dimension(name, tensor):
    print(name+':', np.shape(tensor))