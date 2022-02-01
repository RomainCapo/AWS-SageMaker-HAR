"""Feature engineers the har dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile
import shutil
import boto3
import numpy as np
import pandas as pd
import scipy.io

from sklearn.preprocessing import StandardScaler

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def create_dataset(dataset_dir, subject_index, classes, window, overlap):
    """Create dataset as numpy array format from .mat file

    Args:
        dataset_dir (string): Directory where subjects folder are contained
        subject_index (list): List of subjects index start from 0 for subject 1
        classes (list): List of classes in string
        window (int): Sample length
        overlap (int): Window overlap

    Returns:
        tuple: Input data as numpy array format, Output data as numpy array format, Demarcation index of each subject in the numpy table 
    """
  
    x_data = np.empty((0, window, 4))
    y_data = np.empty((0, 1))  # labels
    subj_inputs = []  # number of inputs for every subject
    tot_rows = 0

    for subject in subject_index:
        subj_inputs.append(0)
    
        for category, name in enumerate(classes):
            matrix_files = os.listdir(f"{dataset_dir}/S{str(subject + 1)}")      
            num_class_files = len([file for file in matrix_files if name in file and file[-4:] == '.mat'])//2

            for record in range(num_class_files):
                acc = scipy.io.loadmat(f'{dataset_dir}/S{subject + 1}/{name}{record + 1}_acc.mat')['ACC']
                ppg = scipy.io.loadmat(f'{dataset_dir}/S{subject + 1}/{name}{record + 1}_ppg.mat')['PPG'][:, 0:2]  # some PPG files have 3 columns instead of 2
                fusion = np.hstack((acc[:, 1:], ppg[:, 1:]))  # remove x axis (time)
                tot_rows += len(fusion)

                # windowing
                # compute number of windows (lazy way)
                i = 0
                num_w = 0
                while i + window  <= len(fusion):
                    i += (window - overlap)
                    num_w += 1
                # compute actual windows
                x_data_part = np.empty((num_w, window, 4))  # preallocate
                i = 0
                for w in range(0, num_w):
                    x_data_part[w] = fusion[i:i + window]
                    i += (window - overlap)
                x_data = np.vstack((x_data, x_data_part))
                y_data = np.vstack((y_data, np.full((num_w, 1), category)))
                subj_inputs[-1] += num_w

    return x_data, y_data, subj_inputs

def get_subjects_index(path):
    """Get subjects index from dataset directory

    Args:
        path (string): Directory where subjects folder are contained

    Returns:
        list: Subjects index list
    """
        
    subjects = []
    for folder in os.listdir(path):
        if folder[0] == 'S' and len(folder)==2:
            subjects.append(int(folder[1])-1)
            
    return subjects

def normalize(x_data):
    """Normalize input data. Subtraction of the mean for the accelerometer components, z-norm for the PPG.  

    Args:
        x_data (np.array): Input data.

    Returns:
        np.array: Normalized data.
    """
    
    for w in x_data:
        # remove mean value from ACC
        w[:, 0] -= np.mean(w[:, 0])  # acc 1
        w[:, 1] -= np.mean(w[:, 1])  # acc 2
        w[:, 2] -= np.mean(w[:, 2])  # acc 3
        # standardize PPG
        w[:, 3] = StandardScaler().fit_transform(w[:, 3].reshape(-1, 1)).reshape((w.shape[0],))  # PPG

    return x_data

def partition_data(subjects, subj_inputs, x_data, y_data):
    """Retrieval of subject data based on subject indices passed in parameters.

    Args:
        subjects (list): List of subjects index.
        subj_inputs (List): List of index subject separation in input data.
        x_data (np.array): Input data
        y_data (np.array): Output data

    Returns:
        tuple: Partionned input data, Partionned output data
    """
        
    # subjects = tuple (0-based)
    x_part = None
    y_part = None
    for subj in subjects:
        skip = sum(subj_inputs[:subj])
        num = subj_inputs[subj]
        xx = x_data[skip : skip + num]
        yy = y_data[skip : skip + num]
        if x_part is None:
            x_part = xx.copy()
            y_part = yy.copy()
        else:
            x_part = np.vstack((x_part, xx))  # vstack creates a copy of the data
            y_part = np.vstack((y_part, yy))
    return x_part, y_part

def clean_data(x_data):
    """Clean input data. Replacement of the Nan values by an interpolated value of the two adjacent points.
    Replacement of zeros values by an interpolated value of the two adjacent points for the PPG.
    Replacement of some missing values by an interpolated value of the two adjacent points for the accelerometer 

    Args:
        x_data (np.array): Cleaned input data
    """
        
    for i in range(x_data.shape[0]):
        for col in range(0,4):
            ids = np.where(np.isnan(x_data[i,:, col]))[0]
            for row in ids:
                x_data[i, row, col] = 0.5 * (x_data[i, row - 1, col] + x_data[i, row + 1, col])

        for col in range(3, 4):
            ids = np.where(x_data[i,:, col] == 0)[0]
            for row in ids:
                x_data[i,row, col] = 0.5 * (x_data[i,row - 1, col] + x_data[i,row + 1, col])

        for col in range(0, 3):
            for row in range(1, x_data.shape[1] - 1):
                if abs(x_data[i,row, col] - x_data[i,row - 1, col]) > 5000 and abs(x_data[i,row, col] - x_data[i,row + 1, col]) > 5000:
                    x_data[i,row, col] = 0.5 * (x_data[i,row - 1, col] + x_data[i,row + 1, col])


def oversampling(x_data, y_data, subj_inputs, num_subjects):
    """Duplicate inputs with classes occurring less, so to have a more balanced distribution.
    We want to do that on a per-subject basis, so to keep subjects separate.
    Moreover, we do that only to the first num_subjects subjects, so to leave test subjects unaltered.

    Args:
        x_data (np.array): Input data.
        y_data (np.array): Output data.
        subj_inputs (List): List of subjects index separation.
        num_subjects (List): List of subjects index separation.

    Returns:
        Tuple: Input data oversampled, Output data oversampled, Corrected subjects index
    """
    
    x_data_over = None
    y_data_over = None
    subj_inputs_over = []
    skip = 0
    for subj_num in subj_inputs[:num_subjects]:
        x_part = x_data[skip : skip + subj_num]
        y_part = y_data[skip : skip + subj_num]
        occurr = (np.sum(y_part == 0), np.sum(y_part == 1), np.sum(y_part == 2))
        assert(occurr[0] == max(occurr))
        mul = (1, occurr[0] // occurr[1], occurr[0] // occurr[2])
        for cl in (1, 2):
            mask = y_part[:, 0] == cl
            x_dup = x_part[mask].copy()
            y_dup = y_part[mask].copy()
            for n in range(0, mul[cl] - 1):
                x_part = np.vstack((x_part, x_dup))
                y_part = np.vstack((y_part, y_dup))
        if x_data_over is None:
            x_data_over = x_part
            y_data_over = y_part
        else:
            x_data_over = np.vstack((x_data_over, x_part))
            y_data_over = np.vstack((y_data_over, y_part))
        subj_inputs_over.append(len(x_part))
        skip += subj_num
    x_data_over = np.vstack((x_data_over, x_data[skip:]))  # subjects not oversampled
    y_data_over = np.vstack((y_data_over, y_data[skip:]))
    subj_inputs_over.extend(subj_inputs[num_subjects:])

    return x_data_over, y_data_over, subj_inputs_over


if __name__ == "__main__":
    logger.info("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--overlap", type=int, required=True)
    parser.add_argument("--data_version", type=str, required=True)
    parser.add_argument("--train_subjects", type=str, required=True)
    parser.add_argument("--validation_subjects", type=str, required=True)
    parser.add_argument("--test_subjects", type=str, required=True)
    parser.add_argument("--classes", type=str, required=True)
    args = parser.parse_args()
    
    train_subjects = tuple(map(int, args.train_subjects.split("-")))
    validation_subjects = tuple(map(int, args.validation_subjects.split("-")))
    test_subjects = tuple(map(int, args.test_subjects.split("-")))
    
    classes = args.classes.split("-")
                        
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/PPG_ACC_dataset.zip"
    s3 = boto3.resource("s3")
    
    if args.data_version == "latest":
        s3.Bucket(bucket).download_file(key, fn)
    else:
        s3.Bucket(bucket).download_file(key, fn, ExtraArgs={'VersionId': args.data_version})
    
    shutil.unpack_archive(fn, f"{base_dir}/data/PPG_ACC_dataset")
    
    DATASET_DIR = base_dir + "/data/PPG_ACC_dataset"
    
    subjects = get_subjects_index(DATASET_DIR) 
    subjects.sort()
    
    logger.info(f"{subjects}")
    
    train_val_subjects = train_subjects + validation_subjects
    
    #If there are more subjects than indicated in the hyperparameters they are added to the training set (in the case of new data for example)
    diff_subjects = set(train_val_subjects + test_subjects) ^ set(subjects)
    if len(diff_subjects) > 0:
        train_val_subjects = train_val_subjects + tuple(diff_subjects)
        
    logger.info(f"Train/validation subjects : {train_val_subjects}")
    logger.info(f"Test subjects : {test_subjects}")
        
    logger.info("Starting create dataset from matrix files.")
    x_data, y_data, subj_inputs = create_dataset(DATASET_DIR, subjects, classes, args.window, args.overlap)
    
    logger.info("Starting data cleaning.")
    clean_data(x_data)
    
    logger.info("Starting data normalization.")
    x_data = normalize(x_data)
    
    logger.info("Starting data oversampling.")
    x_data, y_data, subj_inputs = oversampling(x_data, y_data, subj_inputs, len(train_val_subjects))
    
    logger.info("Starting data partitionning.")
    x_data_train_val, y_data_train_val = partition_data(train_val_subjects, subj_inputs, x_data, y_data)
    x_data_test, y_data_test = partition_data(test_subjects, subj_inputs, x_data, y_data)
    
    logger.info("Save data on bucket.")
    with open(f"{base_dir}/train_val/x_data_train_val.npy", "wb") as f:
        np.save(f, x_data_train_val)
    with open(f"{base_dir}/train_val/y_data_train_val.npy", "wb") as f:
        np.save(f, y_data_train_val)
        
    with open(f"{base_dir}/subjects/subj_inputs.npy", "wb") as f:
        np.save(f, np.array(subj_inputs))
        
    with open(f"{base_dir}/test/x_data_test.npy", "wb") as f:
        np.save(f, x_data_test)
    with open(f"{base_dir}/test/y_data_test.npy", "wb") as f:
        np.save(f, y_data_test)
        
    logger.info("Preprocessing script finished.")
