import os
import json
from datetime import datetime
from typing import List, Tuple
from keras.src.callbacks import History
import pandas as pd
import itertools
import matplotlib.pyplot as plt


class LSTMModelHelper:
    def __init__(self, database_name: str, parameter_list: dict):
        self.db = database_name
        self.params_list = parameter_list


    def get_current_time() -> str:
        """
        Takes no parameters and returns current time in `%Y-%m-%d %H-%M-%S` format.
        """

        try:
            return datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        except Exception as e:
            print("An error occurred when calculating current time: ", e)
            return None
    

    def calculate_times(start_time: int, end_time: int) -> dict:
        """
        Calculate various time-related values based on start and end timestamps.

        Parameters:
        - start_time (int): Start timestamp.
        - end_time (int): End timestamp.

        Returns:
        - dict: Dictionary containing calculated time values.
        """
        assert isinstance(start_time, int), "start_time must be an integer."
        assert isinstance(end_time, int), "end_time must be an integer."

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
        

    def param_combinations(self) -> list:
        """
        Generates all combinations of parameters
        """
        try:
            return list(itertools.product(*self.params_list.values()))
        except Exception as e:
            print("An error occurred when generating combinations: ", e)
            return None
    
    
    def params_to_dict(self) -> List[dict]:
        """
        Returns parameter combinations as list of dict.
        """
        try:
            return [{key: value for key, value in zip(self.params_list.keys(), param)} for param in self.param_combinations(self.params_list)]
        except Exception as e:
            print("An error occurred when converting params tuples to dicts: ", e)
            return None
    

    def params_tuple_to_dict(self, params: tuple) -> dict:
        """
        Convert a single params tuple to a dictionary using parameter names as keys.

        Parameters:
        - params (tuple): The tuple to be converted.
        """
        assert isinstance(params, tuple), 'params must be a tuple.'

        try:
            return {key: value for key, value in zip(self.params_list.keys(), params)}
        except Exception as e:
            print("An error occurred when converting params tuple to dict: ", e)
            return None
    

    def training_data_tuple_to_dict(self, training_data: tuple) -> dict:
        """
        Convert a single training_data tuple to a dictionary using parameter names as keys.

        Parameters:
        - training_data (tuple): The tuple to be converted.
        """
        assert isinstance(training_data, tuple), 'training_data must be a tuple.'

        try:
            return {key: value for key, value in zip(self.training_data_list.keys(), training_data)}
        except Exception as e:
            print("An error occurred when converting tarining_data tuple to dict: ", e)
            return None
    

    def create_folder(folder_name: str, target_path: str = '/Volumes/DATA/LSTM/') -> None:
        """
        Create a folder in the specified path.

        Parameters:
        - folder_name (str): Name of the folder to be created.
        - target_path (str): Path of the folder to be created.
        """
        assert isinstance(folder_name, str), 'folder_name must be a string.'
        assert isinstance(target_path, str), 'target_path must be a string.'

        try:
            # Combine the target path with the provided folder name
            folder_path = os.path.join(target_path, target_path)

            # Create the folder if it doesn't exist
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Folder '{folder_name}' created successfully at: {folder_path}")
            else:
                print(f"Folder '{folder_name}' already exists at: {folder_path}")
        except Exception as e:
            print(f"An error occurred when creating '{folder_name}' folder in {target_path} path: ", e)


    def training_validation_loss_plot(self, training_loss: List[float], validation_loss: List[float], params: Tuple) -> None:
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
            plt.plot(len(training_loss), training_loss, 'r', label='Training loss')
            plt.plot(len(training_loss), validation_loss, 'b', label='Validation loss')
            plt.title(f'Training and validation loss for: {params} (date: {self.get_current_time()})')
            plt.legend(loc=0)
            self.save_image('tra_val_loss', params)
            plt.close()
        except Exception as e:
            print("An error occurred when creating training and validation plot: ", e)


    def close_and_predictions_plot(self, df: pd.DataFrame, close_data: List[float], train_date: List[str], 
                               train_predict: List[float], test_date: List[str], test_predict: List[float], 
                               params: Tuple) -> None:
        """
        Plot close values and predictions.

        Parameters:
        - df (pandas.DataFrame): DataFrame containing date and close data.
        - close_data (List[float]): Original close data.
        - train_date (List[str]): Dates for training data.
        - train_predict (List[float]): Predictions for training data.
        - test_date (List[str]): Dates for test data.
        - test_predict (List[float]): Predictions for test data.
        - params (tuple): Model parameters.
        """
        assert isinstance(df, pd.DataFrame), 'df must be a dictionary.'
        assert isinstance(close_data, list), 'close_data must be a list.'
        assert isinstance(train_date, list), 'train_date must be a list.'
        assert isinstance(train_predict, list), 'train_predict must be a list.'
        assert isinstance(test_date, list), 'test_date must be a list.'
        assert isinstance(test_predict, list), 'test_predict must be a list.'
        assert isinstance(params, str), 'params must be a string.'

        try:
            plt.figure(figsize=(12, 6))
            plt.plot(df['date'], close_data, label='Original Close')
            plt.plot(train_date, train_predict[:,-1], label='Training Predictions')
            plt.plot(test_date, test_predict[:,-1], label='Test Predictions')
            plt.xlabel('Time')
            plt.ylabel('Close Value')
            plt.title(f'Close Values vs. Predictions {params}')
            plt.legend()
            self.save_image('close_and_predictions', params)
            plt.close()
        except Exception as e:
            print("An error occurred when creating close and predictions plot: ", e)


    def future_plot(self, future_dates: List[str], predictions: List[float], params: Tuple) -> None:
        """
        Plot future price predictions.

        Parameters:
        - future_dates (List[str]): Dates for future predictions.
        - predictions (List[float]): Future price predictions.
        - params (tuple): Model parameters.
        """
        assert isinstance(future_dates, list), "future_dates must be a list."
        assert isinstance(predictions, list), "predictions must be a list."
        assert isinstance(params, tuple), "params must be a tuple."

        try:
            plt.figure(figsize=(12, 6))
            plt.plot(future_dates, predictions, label='Predictions')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'Future Price Predictions {params} ({self.get_current_time()})')
            plt.legend()
            self.save_image('future_predictions', params)
            plt.close()
        except Exception as e:
            print("An error occurred when creating future predictions plot: ", e)


    def save_image(self, folder_name: str, params: Tuple) -> None:
        """
        Save the current plot as an image.

        Parameters:
        - folder_name (str): Name of the folder where the image will be saved.
        - params (tuple): Tuple containing parameters used for the plot.
        """

        assert isinstance(folder_name, tuple), "folder_name must be str."
        assert isinstance(params, tuple), "params must be a tuple."

        try:
            self.create_folder(f'images/{folder_name}')
            plt.savefig(f'images/{folder_name}/{folder_name}_{params}.png')
        except Exception as e:
            print(f"An error occurred when saving the {folder_name} plot: {str(e)}")


    def save_data(self, times: dict, history: History, test_loss: list, close_data: list, train_predict: list, test_predict: list, predictions: list, params: dict, table_name: str):
        """
        Saves model data to the database.

        Parameters:
        - Various parameters related to model configuration and training.
        """

        # FIXME - times comes from function way above.
        # Save model data to the database
        params_dict = self.params_tuple_to_dict(params)
        training_data = {
            'start_timestamp': times['start_timestamp'],
            'end_timestamp': times['end_timestamp'],
            'total_time': times['total_time'],
            'start_date': times['start_date'],
            'end_date': times['end_date'],

            'test_loss': test_loss,

            'epoch_used': len(history.history['loss']),
            'training_loss': json.dumps(history.history['loss']),
            'validation_loss': json.dumps(history.history['val_loss']),
            
            'close_data': json.dumps([item for sublist in close_data.values.tolist() for item in sublist]),
            'train_predict': json.dumps(train_predict.tolist()),
            'test_predict': json.dumps(test_predict.tolist()),
            'predictions': json.dumps([item for sublist in predictions.tolist() for item in sublist])
        }

        self.db.save_data(params_dict, training_data, table_name)

