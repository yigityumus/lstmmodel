import os
import json
from datetime import datetime
from typing import List, Tuple
import itertools


from keras.src.callbacks import History
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



class LSTMModelHelper:
    def __init__(self, security: str):
        self.security = security
        self.folder_training_validation_loss = 'training_validation_loss'
        self.folder_close_and_predictions = 'close_and_predictions'
        self.folder_future_predictions = 'future_predictions'

        self.create_folders()

    def create_folders(self) -> None:
        folders = [self.folder_training_validation_loss, self.folder_close_and_predictions, self.folder_future_predictions]
        try:
            self.images_folder = self.create_folder('images', self.security)
        except Exception as e:
            print(f"Error occurred while creating main images folder: {str(e)}")
            return

        for folder in folders:
            try:
                self.create_folder(os.path.join(self.images_folder, folder))
            except Exception as e:
                print(f"Error occurred while creating {folder} folder: {str(e)}")


    @staticmethod
    def create_folder(folder_name: str, target_path: str = "") -> str:
        """
        Create a folder in the specified path.

        Parameters:
        - folder_name (str): Name of the folder to be created.
        - target_path (str): Path of the folder to be created.
        """
        assert isinstance(folder_name, str), f'folder_name must be a string. You provided: {folder_name} : {type(folder_name)}'
        assert isinstance(target_path, str), f'target_path must be a string. You provided: {target_path} : {type(target_path)}'

        try:
            if target_path == "":
                target_path = os.getcwd()

            # Combine the target path with the provided folder name
            folder_path = os.path.join(target_path, folder_name)

            # Create the folder if it doesn't exist
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Folder '{folder_name}' created successfully at: {folder_path}")
            else:
                print(f"Folder '{folder_name}' already exists at: {folder_path}")
            return folder_path
        except Exception as e:
            print(f"An error occurred when creating '{folder_name}' folder in {target_path} path: ", e)
            return None


    @staticmethod
    def get_current_time() -> str:
        """
        Takes no parameters and returns current time in `%Y-%m-%d %H-%M-%S` format.
        """

        try:
            return datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        except Exception as e:
            print("An error occurred when calculating current time: ", e)
            return None
    

    @staticmethod
    def calculate_times(start_time: float, end_time: float) -> dict:
        """
        Calculate various time-related values based on start and end timestamps.

        Parameters:
        - start_time (float): Start timestamp.
        - end_time (float): End timestamp.

        Returns:
        - dict: Dictionary containing calculated time values.
        """
        assert isinstance(start_time, float), "start_time must be an float."
        assert isinstance(end_time, float), "end_time must be an float."

        try:
            # Convert timestamps to datetime objects
            start_date = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
            end_date = datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print("An error occurred when converting timestamps to dates: ", e)
            return None

        try:
            # Calculate total time in seconds
            total_time = end_time - start_time
        except Exception as e:
            print("An error occurred when calculating total time: ", e)
            return None

        return {
            'start_timestamp': start_time,
            'end_timestamp': end_time,
            'total_time': total_time,
            'start_date': start_date,
            'end_date': end_date
        }
        

    @staticmethod
    def param_combinations(params_list: dict) -> list:
        """
        Generates all combinations of parameters
        """
        try:
            return list(itertools.product(*params_list.values()))
        except Exception as e:
            print("An error occurred when generating combinations: ", e)
            return None
    

    @staticmethod
    def params_to_dict(params: tuple, parameter_list: dict) -> List[dict]:
        """
        Returns parameter combinations as list of dict.

        Parameters:
        - params (tuple): The tuple to be converted.
        - parameter_list (dict): The dict of params options.
        """
        assert isinstance(params, tuple), 'params must be a tuple.'
        assert isinstance(parameter_list, dict), 'keys must be a dict.'

        try:
            return {key: value for key, value in zip(parameter_list.keys(), params)}
        except Exception as e:
            print("An error occurred when converting params to dicts: ", e)
            return None


    def plot_training_validation_loss(self, training_loss: List[float], validation_loss: List[float], params: Tuple) -> None:
        """
        Plot training and validation loss.

        Parameters:
        - training_loss (List[float]): List of training loss values.
        - validation_loss (List[float]): List of validation loss values.
        - params (tuple): tuple that contains model parameters.
        """
        assert isinstance(training_loss, list), 'training_loss must be a list.'
        assert isinstance(validation_loss, list), 'validation_loss must be a list.'
        assert isinstance(params, tuple), 'params must be a tuple.'
        assert len(training_loss) == len(validation_loss), 'Lengths of training_loss and validation_loss must be the same.'

        try:        
            plt.figure(figsize=(12, 6))
            plt.plot(training_loss, 'r', label='Training loss')
            plt.plot(validation_loss, 'b', label='Validation loss')
            plt.title(f'Training and validation loss for: {params} (date: {LSTMModelHelper.get_current_time()})')
            plt.legend(loc=0)
            self.save_image(self.folder_training_validation_loss, params)
            plt.close()
        except Exception as e:
            print(f"An error occurred when creating the {self.folder_training_validation_loss} plot: {str(e)}")


    def plot_close_and_predictions(self, df: pd.DataFrame, close_data: pd.DataFrame, train_date: pd.Series, 
                               train_predict: np.ndarray, test_date: pd.Series, test_predict: np.ndarray, 
                               params: Tuple) -> None:
        """
        Plot close values and predictions.

        Parameters:
        - df (pandas.DataFrame): DataFrame containing date and close data.
        - close_data (pandas.DataFrame): Original close data.
        - train_date (pandas.DataFrame): Dates for training data.
        - train_predict (numpy.ndarray): Predictions for training data.
        - test_date (pandas.DataFrame): Dates for test data.
        - test_predict (numpy.ndarray): Predictions for test data.
        - params (tuple): Model parameters.
        """
        assert isinstance(df, pd.DataFrame), 'df must be a pandas.DataFrame.'
        assert isinstance(close_data, pd.DataFrame), 'close_data must be a pandas.DataFrame.'
        assert isinstance(train_date, pd.Series), 'train_date must be a pandas.Series.'
        assert isinstance(train_predict, np.ndarray), 'train_predict must be a numpy.ndarray.'
        assert isinstance(test_date, pd.Series), 'test_date must be a pandas.Series.'
        assert isinstance(test_predict, np.ndarray), 'test_predict must be a numpy.ndarray.'
        assert isinstance(params, tuple), 'params must be a tuple.'

        try:
            plt.figure(figsize=(12, 6))
            plt.plot(df['date'], close_data, label='Original Close')
            plt.plot(train_date, train_predict[:,-1], label='Training Predictions')
            plt.plot(test_date, test_predict[:,-1], label='Test Predictions')
            plt.xlabel('Time')
            plt.ylabel('Close Value')
            plt.title(f'Close Values vs. Predictions {params}')
            plt.legend()
            self.save_image(self.folder_close_and_predictions, params)
            plt.close()
        except Exception as e:
            print(f"An error occurred when creating {self.folder_close_and_predictions} plot: {str(e)}")


    def plot_future_predictions(self, future_dates: pd.DatetimeIndex, predictions: np.ndarray, params: Tuple) -> None:
        """
        Plot future price predictions.

        Parameters:
        - future_dates (pandas.DatetimeIndex): Dates for future predictions.
        - predictions (numpy.ndarray): Future price predictions.
        - params (tuple): Model parameters.
        """
        assert isinstance(future_dates, pd.DatetimeIndex), "future_dates must be a pandas.DatetimeIndex."
        assert isinstance(predictions, np.ndarray), "predictions must be a numpy.ndarray."
        assert isinstance(params, tuple), "params must be a tuple."

        try:
            plt.figure(figsize=(12, 6))
            plt.plot(future_dates, predictions, label='Predictions')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'Future Price Predictions {params} ({LSTMModelHelper.get_current_time()})')
            plt.legend()
            self.save_image(self.folder_future_predictions, params)
            plt.close()
        except Exception as e:
            print(f"An error occurred when creating {self.folder_future_predictions} plot: {str(e)}")


    def save_image(self, folder_name, params: Tuple) -> None:
        """
        Save the current plot as an image.

        Parameters:
        - folder_name (str): Name of the folder where the image will be saved.
        - params (tuple): Tuple containing parameters used for the plot.
        """

        assert isinstance(folder_name, str), f"folder_name must be str. You provided: {folder_name}. Type: {type(folder_name)}"
        assert isinstance(params, tuple), f"params must be a tuple. You provided type: {type(folder_name)}"

        try:
            # images_folder = os.path.join(self.security, 'images')
            # LSTMModelHelper.create_folder(images_folder)
            # LSTMModelHelper.create_folder(os.path.join(images_folder, folder_name))
            file_path = os.path.join(self.images_folder, folder_name, f"{folder_name}_{str(params)}.png")
            plt.savefig(file_path)
        except Exception as e:
            print(f"An error occurred when saving the {folder_name} plot: {str(e)}")


    @staticmethod
    def save_data(times: dict, history: History, test_loss: list, close_data: pd.DataFrame, train_predict: np.ndarray, test_predict: np.ndarray, predictions: np.ndarray, params: dict, table_name: str) -> dict:
        """
        Saves model data to the database.

        Parameters:
        - Various parameters related to model configuration and training.
        """

        data = {
            'params_dict': params,
            'training_data': {
                'start_timestamp': times['start_timestamp'],
                'end_timestamp': times['end_timestamp'],
                'total_time': times['total_time'],
                'start_date': times['start_date'],
                'end_date': times['end_date'],
                'epoch_used': len(history.history['loss']),
                'test_loss': test_loss,

                'training_loss': json.dumps(history.history['loss']),
                'validation_loss': json.dumps(history.history['val_loss']),
                'close_data': json.dumps([item for sublist in close_data.values.tolist() for item in sublist]),
                'train_predict': json.dumps(train_predict.tolist()),
                'test_predict': json.dumps(test_predict.tolist()),
                'predictions': json.dumps([item for sublist in predictions.tolist() for item in sublist])
            },
            'table_name': table_name
        }

        return data
