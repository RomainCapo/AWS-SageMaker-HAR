"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import pickle
import tarfile
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
def create_json_conf_matrix(conf_matrix):
    """Convert numpy confusion matrix in JSON format.

    Args:
        conf_matrix (np.array): Confusion Matrix

    Returns:
        dict: Confusion matrix as JSON.
    """
    
    json_conf_matrix = {}
    for i in range(conf_matrix.shape[0]):
        json_conf_matrix[f"{i}"] = {}
        for j in range(conf_matrix.shape[1]):
            json_conf_matrix[f"{i}"][f"{j}"] = int(conf_matrix[i,j])

    return json_conf_matrix 

if __name__ == "__main__":
    
    install("tensorflow==2.5")
    from tensorflow.keras import models
    
    logger.info("Starting evaluation.")
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path, "r:gz") as tar:
        tar.extractall("./model")
        
    model = models.load_model("./model/1")

    logger.info("Reading test data.")
    test_path = "/opt/ml/processing/test"
    x_data_test = np.load(os.path.join(test_path, 'x_data_test.npy'))
    y_data_test = np.load(os.path.join(test_path, 'y_data_test.npy'))
    
    logger.info("Performing predictions against test data.")
    y_pred_proba = model.predict(x_data_test)
    y_pred = np.argmax(y_pred_proba,axis=1)
    
    logger.info(f"Model evaluate : {model.evaluate(x_data_test, y_data_test)}")
    
    logger.info("Calculating metrics.")
    accuracy = accuracy_score(y_data_test, y_pred)
    precision = precision_score(y_data_test, y_pred, average='weighted')
    recall = recall_score(y_data_test, y_pred, average='weighted')
    f1 = f1_score(y_data_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_data_test, y_pred)
    
    logger.info(f"accuracy : {accuracy}")
    logger.info(f"precision : {precision}")
    logger.info(f"recall : {recall}")
    logger.info(f"f1 : {f1}")
    
    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy" : {
              "value" : accuracy,"standard_deviation" : "Nan"
              },
            "weighted_precision" : {
              "value" : precision,"standard_deviation" : "Nan"
              },
            "weighted_recall" : {
              "value" : recall,"standard_deviation" : "Nan"
             },
            "weighted_f1" : {
              "value" : f1,"standard_deviation" : "Nan"
            },
            "confusion_matrix":create_json_conf_matrix(conf_matrix)
        }
    }
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with accuracy: %f", accuracy)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))