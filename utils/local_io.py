# This file is used for reading dcm files
from utils.basic import *
import numpy as np
import nibabel as nib
import csv
import scipy.io as scio


def save_nii(dcm_matrix, affine_matrix, file_name, save_dir):
    nii_img = nib.Nifti1Image(dcm_matrix, affine_matrix)
    file_name = file_name
    nib.save(nii_img, join_path(save_dir, file_name))
    
def save_csv(file, data):
    with open(file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)
    print('writing csv file in: %s' % file)


def load_csv(split_file):
    with open(split_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        data_list = {}
        for item in reader:
            data_list[item[0]] = item[1:]
    return data_list

def read_mat(file_path):
    return scio.loadmat(file_path)

# data_new dict into mat files for further use
def save_mat(mat_data, case_name, save_dir):
    file = join_path(save_dir, case_name)
    for key, value in mat_data.items():
        if value is None:
            raise ValueError('Find empty content in case_%s:%s' % (case_name, key))
    scio.savemat(file, mat_data, format='5', do_compression=True)

