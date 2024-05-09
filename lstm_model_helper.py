import os
from datetime import datetime
from typing import List, Dict, Any


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import ast



class LSTMPlotter:
    def __init__(self):
        pass


    @staticmethod
    def get_current_time() -> str:
        """
        Takes no parameters and returns current time in `%Y-%m-%d %H-%M-%S` format.
        """

        try:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print("An error occurred when calculating current time: ", e)
            return None
    

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
    

    def plot_residuals(self, folder_path: str, model_type: str, index: int, params_str: str, predictions: np.ndarray, residuals: np.ndarray, residuals_percentage: np.ndarray) -> None:
        assert isinstance(folder_path, str), "folder_path must be a str."
        assert isinstance(model_type, str), "model_type must be a str."
        assert isinstance(index, int), "index must be a int."
        assert isinstance(params_str, str), "params_str must be a str."

        try:
            plot_type = 'residuals'
            # predictions = ast.literal_eval(predictions)
            # residuals = ast.literal_eval(residuals)
            # residuals_percentage = ast.literal_eval(residuals_percentage)

            # Create a plot
            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Scatter plot for residuals
            ax1.scatter(predictions, residuals, color='b', label='Residuals')
            ax1.set_xlabel('Predictions')
            ax1.set_ylabel('Residuals', color='b')

            # Calculate the middle value of the y-axis range
            middle_value = np.mean(ax1.get_ylim())

            # Add a dotted line at the middle of the y-axis
            ax1.axhline(y=middle_value, color='gray', linestyle='--')

            # Create a second y-axis displaying percentage values
            ax2 = ax1.twinx()
            ax2.scatter(predictions, residuals_percentage, color='r', label='Residuals Percentage')
            ax2.set_ylabel('Residuals Percentage', color='r')

            plt.title(f'{model_type} Residuals - {index} {params_str} ({LSTMPlotter.get_current_time()})')
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


