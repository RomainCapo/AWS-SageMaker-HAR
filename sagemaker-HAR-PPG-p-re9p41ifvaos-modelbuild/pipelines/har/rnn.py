import argparse
import os
import numpy as np
import logging
import tensorflow as tf

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, BatchNormalization, LSTM, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


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

def create_model(args, num_classes, num_features):
    """Create LSTM model.

    Args:
        args (argparse): Argparse arguments objects.
        num_classes (int): Number of classes.
        num_features (int): Number of input features

    Returns:
        tuple: Partionned input data, Partionned output data
    """
        
    model = Sequential()
    
    model.add(Input(shape=(args.window, num_features)))
    model.add(Dense(args.num_cell_dense1, name='dense1'))
    model.add(BatchNormalization(name='norm'))
    
    model.add(LSTM(args.num_cell_lstm1, return_sequences=True, name='lstm1'))
    model.add(Dropout(args.dropout_rate, name='drop2'))
    
    model.add(LSTM(args.num_cell_lstm2, name='lstm2'))
    model.add(Dropout(args.dropout_rate, name='drop3'))
    
    model.add(Dense(num_classes, name='dense2')) 
    
    optimizer = Adam(learning_rate=args.learning_rate)
    
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizer, metrics=['accuracy'])
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    logger.debug(f"Starting parses arguments.")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_cell_dense1', type=int, default=32)
    parser.add_argument('--num_cell_lstm1', type=int, default=32)
    parser.add_argument('--num_cell_lstm2', type=int, default=32)
    parser.add_argument('--num_cell_lstm3', type=int, default=32)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    
    parser.add_argument('--train_val', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_VAL'))
    parser.add_argument('--subjects', type=str, default=os.environ.get('SM_CHANNEL_SUBJECTS'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    parser.add_argument('--output_path', type=str, default=os.environ.get('SM_OUTPUT_PATH'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--overlap", type=int, required=True)
    parser.add_argument("--train_subjects", type=str, required=True)
    parser.add_argument("--validation_subjects", type=str, required=True)
    parser.add_argument("--test_subjects", type=str, required=True)
    parser.add_argument("--classes", type=str, required=True)

    
    args = parser.parse_args()
    
    train_subjects = tuple(map(int, args.train_subjects.split("-")))
    validation_subjects = tuple(map(int, args.validation_subjects.split("-")))
    test_subjects = tuple(map(int, args.test_subjects.split("-")))
    
    logger.info(f"GPU device: {tf.test.gpu_device_name()}")
    print(f"GPU device: {tf.test.gpu_device_name()}")
    
    logger.info(f"Training data loading.")
    x_data_train_val = np.load(os.path.join(args.train_val, 'x_data_train_val.npy'))
    y_data_train_val = np.load(os.path.join(args.train_val, 'y_data_train_val.npy'))
    subj_inputs = np.load(os.path.join(args.subjects, 'subj_inputs.npy'))
    
    # If there are more subjects than indicated in the hyperparameters they are added to the training set (in the case of new data for example)
    diff_subjects = abs(len(set(train_subjects + validation_subjects + test_subjects)) - len(subj_inputs))
    if diff_subjects != 0:
        train_subjects = train_subjects + tuple(range(len(subj_inputs) - diff_subjects, len(subj_inputs)))
        
    logger.info(f"Train subjects : {train_subjects}")
    logger.info(f"Validation subjects : {validation_subjects}")
    logger.info(f"Test subjects : {test_subjects}")
                            
    x_data_train, y_data_train = partition_data(train_subjects, subj_inputs, x_data_train_val, y_data_train_val)
    x_data_val, y_data_val = partition_data(validation_subjects, subj_inputs, x_data_train_val, y_data_train_val)
    
    num_classes = len(set(y_data_train_val.flatten()))
    num_features = x_data_train_val.shape[2]
    
    logger.info(f"{num_classes}")
    logger.info(f"{num_features}")
    
    logger.info(f"Model creation.")
    model = create_model(args, num_classes, num_features)

    logger.info("Staring model training.")
    history = model.fit(x_data_train, y_data_train, epochs=args.epochs, validation_data=(x_data_val, y_data_val))
    
    logger.debug(f"Output path {args.output_path}")
    logger.debug(f"Model dir {args.model_dir}")

    model.save(args.model_dir + '/1')
    logger.debug("Training script finished.")





