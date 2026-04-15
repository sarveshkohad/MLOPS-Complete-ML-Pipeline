import os
import logging
import pandas as pd
import numpy as np
import pickle 
from sklearn.ensemble import RandomForestClassifier

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.logs')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s -%(name)s -%(levelname)s -%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str):
    """"
    Load data from a csv file
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from: %s with shape: %s", file_path, df.shape)
        return df
    except pd.errors.ParserError:
        logger.error("Failed to parse csv file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading the data: %s", e)
        raise

def train_model(X_train:np.ndarray, y_train:np.ndarray, params:dict):
    """
    Train the Random forest model
    :param X_train - training features
    :param y_train - training labels
    :param params - dictionary of Hyperparameters
    :return - Trained RandomForestClassifier
    """

    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be same")
        
        logger.debug("Initializing RandomForest model with parameters: %s", params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        logger.debug("Model training started with %d samples ", X_train.shape[0])
        logger.debug("NaNs in X_train: %s", np.isnan(X_train).sum())
        logger.debug("NaNs in y_train: %s", np.isnan(y_train).sum())
        clf.fit(X_train, y_train)
        logger.debug("Model training completed")

        return clf
    except ValueError as e:
        logger.error("ValueError during model training %s ", e)
        raise
    except Exception as e:
        logger.error('Error during model trainig: %s', e)
        raise

def save_model(model, file_path):
    """
    Save the trained model to a file
    :param model: Trained model object
    :param file_path: Path to save the model file
    """

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug("Model saved to %s", file_path)
    except FileNotFoundError as e:
        logger.error("File path not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error while saving the model: %s", e)
        raise

def main():
    try:
        params = {'n_estimators': 25, 'random_state': 2}
        train_data = load_data("./data/processed/train_tfidf.csv")
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values

        clf = train_model(X_train, y_train, params)

        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error("Failed to complete the model building process: %s", e)

if __name__ == "__main__":
    main()