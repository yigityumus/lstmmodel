import os
from datetime import datetime
from typing import List, Dict, Any


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np



class LSTMPlotter:
    def __init__(self):
        pass


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
    

    # def plot_training_validation_loss(self, index: int, model_type: str, training_loss: List[float], validation_loss: List[float], params: Dict[str, Any]) -> None:
    #     """
    #     Plot training and validation loss.

    #     Parameters:
    #     - training_loss (List[float]): List of training loss values.
    #     - validation_loss (List[float]): List of validation loss values.
    #     - params (tuple): tuple that contains model parameters.
    #     """
    #     assert isinstance(training_loss, list), 'training_loss must be a list.'
    #     assert isinstance(validation_loss, list), 'validation_loss must be a list.'
    #     assert isinstance(params, dict), 'params must be a dict.'
    #     assert len(training_loss) == len(validation_loss), 'Lengths of training_loss and validation_loss must be the same.'

    #     try:
    #         params_str = LSTMModelHelper.dict_to_tuple(params)        
    #         plt.figure(figsize=(12, 6))
    #         plt.plot(training_loss, 'r', label='Training loss')
    #         plt.plot(validation_loss, 'b', label='Validation loss')
    #         plt.title(f'{model_type} Training and validation loss - {index} {params_str} (date: {LSTMModelHelper.get_current_time()})')
    #         plt.legend(loc=0)
    #         self.save_image(index, model_type, self.folder_training_validation_loss, params)
    #         plt.close()
    #     except Exception as e:
    #         print(f"An error occurred when creating the {self.folder_training_validation_loss} plot: {str(e)}")

    # def plot_close_and_predictions(self, index: int, model_type: str, df: pd.DataFrame, close_data: pd.DataFrame, train_date: pd.Series, 
    #                            train_predict: np.ndarray, test_date: pd.Series, test_predict: np.ndarray, 
    #                            params: Dict[str, Any]) -> None:
    #     """
    #     Plot close values and predictions.

    #     Parameters:
    #     - df (pandas.DataFrame): DataFrame containing date and close data.
    #     - close_data (pandas.DataFrame): Original close data.
    #     - train_date (pandas.DataFrame): Dates for training data.
    #     - train_predict (numpy.ndarray): Predictions for training data.
    #     - test_date (pandas.DataFrame): Dates for test data.
    #     - test_predict (numpy.ndarray): Predictions for test data.
    #     - params (Dict[str, Any]): Model parameters.
    #     """
    #     assert isinstance(df, pd.DataFrame), 'df must be a pandas.DataFrame.'
    #     assert isinstance(close_data, pd.DataFrame), 'close_data must be a pandas.DataFrame.'
    #     assert isinstance(train_date, pd.Series), 'train_date must be a pandas.Series.'
    #     assert isinstance(train_predict, np.ndarray), 'train_predict must be a numpy.ndarray.'
    #     assert isinstance(test_date, pd.Series), 'test_date must be a pandas.Series.'
    #     assert isinstance(test_predict, np.ndarray), 'test_predict must be a numpy.ndarray.'
    #     assert isinstance(params, dict), 'params must be a dict.'

    #     try:
    #         fig, ax = plt.subplots(figsize=(12, 6))
    #         # plt.figure(figsize=(12, 6))
    #         plt.plot(df['date'], close_data, label='Original Close')
    #         plt.plot(train_date, train_predict[:,-1], label='Training Predictions')
    #         plt.plot(test_date, test_predict[:,-1], label='Test Predictions')

    #         # Use YearLocator and YearFormatter on the axes object (ax)
    #         ax.xaxis.set_major_locator(mdates.YearLocator())
    #         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    #         params_str = LSTMModelHelper.dict_to_tuple(params) 
    #         plt.xlabel('Time')
    #         plt.ylabel('Close Value')
    #         plt.title(f'{model_type} Close Values vs. Predictions - {index} {params_str}')
    #         plt.legend()
    #         self.save_image(index, model_type, self.folder_close_and_predictions, params)
    #         plt.close()
    #     except Exception as e:
    #         print(f"An error occurred when creating {self.folder_close_and_predictions} plot: {str(e)}")

    # def plot_future_predictions(self, index: int, model_type: str, future_dates: pd.DatetimeIndex, predictions: np.ndarray, params: Dict[str, Any]) -> None:
    #     """
    #     Plot future price predictions.

    #     Parameters:
    #     - future_dates (pandas.DatetimeIndex): Dates for future predictions.
    #     - predictions (numpy.ndarray): Future price predictions.
    #     - params (Dict[str, Any]): Model parameters.
    #     """
    #     assert isinstance(future_dates, pd.DatetimeIndex), "future_dates must be a pandas.DatetimeIndex."
    #     assert isinstance(predictions, np.ndarray), "predictions must be a numpy.ndarray."
    #     assert isinstance(params, dict), "params must be a dict."

    #     try:
    #         params_str = LSTMModelHelper.dict_to_tuple(params)
    #         plt.figure(figsize=(12, 6))
    #         plt.plot(future_dates, predictions, label='Predictions')
    #         plt.xlabel('Date')
    #         plt.ylabel('Price')
    #         plt.title(f'{model_type} Future Price Predictions - {index} {params_str} ({LSTMModelHelper.get_current_time()})')
    #         plt.legend()
    #         self.save_image(index, model_type, self.folder_future_predictions, params)
    #         plt.close()
    #     except Exception as e:
    #         print(f"An error occurred when creating {self.folder_future_predictions} plot: {str(e)}")

    # def save_image(self, index: int, model_type: str, folder_name, params: dict) -> None:
    #     """
    #     Save the current plot as an image.

    #     Parameters:
    #     - folder_name (str): Name of the folder where the image will be saved.
    #     - params (dict): Dictionary object containing parameters used for the plot.
    #     """

    #     assert isinstance(folder_name, str), f"folder_name must be str. You provided: {folder_name}. Type: {type(folder_name)}"
    #     assert isinstance(params, dict), f"params must be a dict. You provided type {type(params)}"

    #     try:
    #         # images_folder = os.path.join(self.ticker_fullname, 'images')
    #         # LSTMModelHelper.create_folder(images_folder)
    #         # LSTMModelHelper.create_folder(os.path.join(images_folder, folder_name))
    #         params_str = LSTMModelHelper.dict_to_tuple(params)
    #         file_path = os.path.join(self.images_folder, folder_name, f"{model_type}_{folder_name}_{index}_{str(params_str)}.png")
    #         plt.savefig(file_path)
    #     except Exception as e:
    #         print(f"An error occurred when saving the {folder_name} plot: {str(e)}")
    

    def plot_training_validation_loss(self, folder_path: str, model_type: str, index: int, params_str: str, training_loss: List[float], validation_loss: List[float]) -> None:
        assert isinstance(folder_path, str), 'folder_path must be a str.'
        assert isinstance(model_type, str), 'model_type must be a str.'
        assert isinstance(index, int), 'index must be a int.'
        assert isinstance(params_str, str), 'params_str must be a str.'
        assert isinstance(training_loss, list), 'training_loss must be a list.'
        assert isinstance(validation_loss, list), 'validation_loss must be a list.'
        assert len(training_loss) == len(validation_loss), 'Lengths of training_loss and validation_loss must be the same.'

        try:
            plot_type = 'training_validation_loss'        
            plt.figure(figsize=(12, 6))
            plt.plot(training_loss, 'r', label='Training loss')
            plt.plot(validation_loss, 'b', label='Validation loss')
            plt.title(f'{model_type} Training and validation loss - {index} {params_str} ({LSTMPlotter.get_current_time()})')
            plt.legend(loc=0)
            self.save_image(folder_path, model_type, index, params_str, plot_type)
            plt.close()
        except Exception as e:
            print(f"An error occurred when creating the {plot_type} plot: {str(e)}")
    

    def plot_close_and_predictions(self, folder_path: str, model_type: str, index: int, params_str: str, df: pd.DataFrame, close_data: pd.DataFrame, train_date: pd.Series, 
                               train_predict: np.ndarray, test_date: pd.Series, test_predict: np.ndarray) -> None:
        assert isinstance(folder_path, str), 'folder_path must be a str.'
        assert isinstance(model_type, str), 'model_type must be a str.'
        assert isinstance(index, int), 'index must be a int.'
        assert isinstance(params_str, str), 'params_str must be a str.'
        assert isinstance(df, pd.DataFrame), 'df must be a pandas.DataFrame.'
        assert isinstance(close_data, pd.DataFrame), 'close_data must be a pandas.DataFrame.'
        assert isinstance(train_date, pd.Series), 'train_date must be a pandas.Series.'
        assert isinstance(train_predict, np.ndarray), 'train_predict must be a numpy.ndarray.'
        assert isinstance(test_date, pd.Series), 'test_date must be a pandas.Series.'
        assert isinstance(test_predict, np.ndarray), 'test_predict must be a numpy.ndarray.'

        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            # plt.figure(figsize=(12, 6))
            plt.plot(df['date'], close_data, label='Original Close')
            plt.plot(train_date, train_predict[:,-1], label='Training Predictions')
            plt.plot(test_date, test_predict[:,-1], label='Test Predictions')

            # Use YearLocator and YearFormatter on the axes object (ax)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            plot_type = 'close_and_predictions'
            plt.xlabel('Time')
            plt.ylabel('Close Value')
            plt.title(f'{model_type} Close Values vs. Predictions - {index} {params_str} ({LSTMPlotter.get_current_time()})')
            plt.legend()
            self.save_image(folder_path, model_type, index, params_str, plot_type)
            plt.close()
        except Exception as e:
            print(f"An error occurred when creating {plot_type} plot: {str(e)}")
    

    def plot_future_predictions(self, folder_path: str, model_type: str, index: int, params_str: str, future_dates: pd.DatetimeIndex, predictions: np.ndarray) -> None:
        assert isinstance(folder_path, str), "folder_path must be a str."
        assert isinstance(model_type, str), "model_type must be a str."
        assert isinstance(index, int), "index must be a int."
        assert isinstance(params_str, str), "params_str must be a str."
        assert isinstance(future_dates, pd.DatetimeIndex), "future_dates must be a pandas.DatetimeIndex."
        assert isinstance(predictions, np.ndarray), "predictions must be a numpy.ndarray."

        try:
            plot_type = 'future_predictions'
            plt.figure(figsize=(12, 6))
            plt.plot(future_dates, predictions, label='Predictions')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'{model_type} Future Price Predictions - {index} {params_str} ({LSTMPlotter.get_current_time()})')
            plt.legend()
            self.save_image(folder_path, model_type, index, params_str, plot_type)
            plt.close()
        except Exception as e:
            print(f"An error occurred when creating {plot_type} plot: {str(e)}")


    def save_image(self, folder_path: str, model_type: str, index: int, params_str: str, plot_type: str) -> None:
        assert isinstance(folder_path, str), f"folder_path must be str. You provided: {folder_path}. Type: {type(folder_path)}"
        assert isinstance(model_type, str), f"model_type must be str. You provided: {model_type}. Type: {type(model_type)}"
        assert isinstance(index, int), f"index must be int. You provided: {index}. Type: {type(index)}"
        assert isinstance(params_str, str), f"params_str must be a str. You provided type {type(params_str)}"

        try:
            file_path = os.path.join(folder_path, f"{model_type}_{plot_type}_{index}_{params_str}.png")
            plt.savefig(file_path)
        except Exception as e:
            print(f"An error occurred when saving the {plot_type} plot: {str(e)}")


