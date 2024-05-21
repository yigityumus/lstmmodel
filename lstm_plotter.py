import os
from datetime import datetime
from typing import List
import logging

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from typeguard import typechecked


class LSTMPlotter:
    def __init__(self):
        pass

    @staticmethod
    def get_current_time() -> str:
        """
        Returns the current time in `%Y-%m-%d %H:%M:%S` format.
        """
        try:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logging.error("Error occurred when calculating current time: %s", e)
            return ""

    @staticmethod
    @typechecked
    def plot_training_validation_loss(
        folder_path: str, model_type: str, index: int, params_str: str, 
        training_loss: List[float], validation_loss: List[float]
    ) -> None:
        """
        Plots training and validation loss.

        Args:
            folder_path (str): Path to save the plot.
            model_type (str): Type of the model.
            index (int): Index for the plot.
            params_str (str): Parameters string for the plot.
            training_loss (List[float]): Training loss values.
            validation_loss (List[float]): Validation loss values.
        """
        LSTMPlotter._validate_losses(training_loss, validation_loss)

        try:
            plt.figure(figsize=(12, 6))
            plt.plot(training_loss, 'r', label='Training loss')
            plt.plot(validation_loss, 'b', label='Validation loss')
            plt.title(f'{model_type} Training and Validation Loss - {index} {params_str} ({LSTMPlotter.get_current_time()})')
            plt.legend(loc=0)
            LSTMPlotter.save_image(folder_path, model_type, index, params_str, 'training_validation_loss')
            plt.close()
        except Exception as e:
            logging.error("Error occurred when creating the training_validation_loss plot: %s", e)
            plt.close()

    @staticmethod
    @typechecked
    def plot_close_and_predictions(
        folder_path: str, model_type: str, index: int, params_str: str, 
        df: pd.DataFrame, close_data: pd.DataFrame, train_date: pd.Series, 
        train_predict: np.ndarray, test_date: pd.Series, test_predict: np.ndarray
    ) -> None:
        """
        Plots close values and predictions.

        Args:
            folder_path (str): Path to save the plot.
            model_type (str): Type of the model.
            index (int): Index for the plot.
            params_str (str): Parameters string for the plot.
            df (pd.DataFrame): Dataframe containing dates.
            close_data (pd.DataFrame): Dataframe containing close values.
            train_date (pd.Series): Series of training dates.
            train_predict (np.ndarray): Array of training predictions.
            test_date (pd.Series): Series of test dates.
            test_predict (np.ndarray): Array of test predictions.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['date'], close_data, label='Original Close')
            ax.plot(train_date, train_predict[:, -1], label='Training Predictions')
            ax.plot(test_date, test_predict[:, -1], label='Test Predictions')

            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            plt.xlabel('Time')
            plt.ylabel('Close Value')
            plt.title(f'{model_type} Close Values vs. Predictions - {index} {params_str} ({LSTMPlotter.get_current_time()})')
            plt.legend()
            LSTMPlotter.save_image(folder_path, model_type, index, params_str, 'close_and_predictions')
            plt.close()
        except Exception as e:
            logging.error("Error occurred when creating close_and_predictions plot: %s", e)
            plt.close()

    @staticmethod
    @typechecked
    def plot_future_predictions(
        folder_path: str, model_type: str, index: int, params_str: str, 
        future_dates: pd.DatetimeIndex, control_data: np.ndarray, predictions: np.ndarray
    ) -> None:
        """
        Plots future price predictions.

        Args:
            folder_path (str): Path to save the plot.
            model_type (str): Type of the model.
            index (int): Index for the plot.
            params_str (str): Parameters string for the plot.
            future_dates (pd.DatetimeIndex): Datetime index for future dates.
            control_data (np.ndarray): Array of control data.
            predictions (np.ndarray): Array of predictions.
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(future_dates, control_data, label='Original Data', color='blue')
            plt.plot(future_dates, predictions, label='Predictions', color='orange')

            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'{model_type} Future Price Predictions - {index} {params_str} ({LSTMPlotter.get_current_time()})')
            plt.legend()
            LSTMPlotter.save_image(folder_path, model_type, index, params_str, 'future_predictions')
            plt.close()
        except Exception as e:
            logging.error("Error occurred when creating future_predictions plot: %s", e)
            plt.close()

    @staticmethod
    @typechecked
    def plot_residuals(
        folder_path: str, security: str, model_type: str, index: int, params_str: str, 
        predictions: np.ndarray, residuals: np.ndarray, residuals_percentage: np.ndarray
    ) -> None:
        """
        Plots residuals and residuals percentage.

        Args:
            folder_path (str): Path to save the plot.
            security (str): Security name.
            model_type (str): Type of the model.
            index (int): Index for the plot.
            params_str (str): Parameters string for the plot.
            predictions (np.ndarray): Array of predictions.
            residuals (np.ndarray): Array of residuals.
            residuals_percentage (np.ndarray): Array of residuals percentage.
        """
        try:
            fig, ax1 = plt.subplots(figsize=(12, 6))

            ax1.scatter(predictions, residuals, color='b', label='Residuals')
            ax1.set_xlabel(f' {security} Predictions ($)')
            ax1.set_ylabel(f' {security} Residuals ($)', color='b')

            middle_value = np.mean(ax1.get_ylim())
            ax1.axhline(y=middle_value, color='gray', linestyle='--')

            ax2 = ax1.twinx()
            ax2.scatter(predictions, residuals_percentage, color='r', label='Residuals Percentage')
            ax2.set_ylabel('Residuals Percentage (%)', color='r')

            plt.title(f'{model_type} Residuals - {index} {params_str} ({LSTMPlotter.get_current_time()})')
            LSTMPlotter.save_image(folder_path, model_type, index, params_str, 'residuals')
            plt.close()
        except Exception as e:
            logging.error("Error occurred when creating residuals plot: %s", e)
            plt.close()

    @staticmethod
    @typechecked
    def save_image(folder_path: str, model_type: str, index: int, params_str: str, plot_type: str) -> None:
        """
        Saves the plot image to the specified folder.

        Args:
            folder_path (str): Path to save the image.
            model_type (str): Type of the model.
            index (int): Index for the plot.
            params_str (str): Parameters string for the plot.
            plot_type (str): Type of the plot.
        """
        try:
            file_path = os.path.join(folder_path, f"{model_type}_{plot_type}_{index}_{params_str}.png")
            plt.savefig(file_path)
        except Exception as e:
            logging.error("Error occurred when saving the %s plot: %s", plot_type, e)

    @staticmethod
    @typechecked
    def _validate_losses(training_loss: List[float], validation_loss: List[float]) -> None:
        """
        Validates the lengths of training and validation losses.

        Args:
            training_loss (List[float]): Training loss values.
            validation_loss (List[float]): Validation loss values.

        Raises:
            ValueError: If lengths of training and validation losses do not match.
        """
        if len(training_loss) != len(validation_loss):
            raise ValueError("Lengths of training_loss and validation_loss must be the same.")


# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
