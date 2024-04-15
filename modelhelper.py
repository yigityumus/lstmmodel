import os
import json
import matplotlib.pyplot as plt
from datetime import datetime


# TODO - Remove this file once you complete the other TODOs


class LSTMModelHelper:
    def __init__(self, database_name: str):
        self.db = database_name

    def training_validation_loss_plot(self, history, params, current_time):
        """
        Plots training and validation loss.

        Parameters:
        - history: Training history object containing loss and validation loss.
        - params: Model parameters.
        - current_time: Current timestamp.
        """
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title(f'Training and validation loss for: {params} (date: {current_time})')
        plt.legend(loc=0)
        self.save_plot(params, 'tra_val_loss', current_time)

    def close_and_predictions_plot(self, df, close_data, train_date, train_predict, test_date, test_predict, params):
        """
        Plots close values and predictions.

        Parameters:
        - df: DataFrame containing date and close data.
        - close_data: Original close data.
        - train_date: Dates for training data.
        - train_predict: Predictions for training data.
        - test_date: Dates for test data.
        - test_predict: Predictions for test data.
        - params: Model parameters.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], close_data, label='Original Close')
        plt.plot(train_date, train_predict[:,-1], label='Training Predictions')
        plt.plot(test_date, test_predict[:,-1], label='Test Predictions')
        plt.xlabel('Time')
        plt.ylabel('Close Value')
        plt.title(f'Close Values vs. Predictions {params}')
        plt.legend()
        self.save_plot(params, 'close_training_test', '')

    def future_plot(self, future_dates, predictions, params, current_time):
        """
        Plots future price predictions.

        Parameters:
        - future_dates: Dates for future predictions.
        - predictions: Future price predictions.
        - params: Model parameters.
        - current_time: Current timestamp.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(future_dates, predictions, label='Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'Future Price Predictions {params} ({current_time})')
        plt.legend()
        self.save_plot(params, 'future_predictions', current_time)

    def save_plot(self, params, plot_type, current_time):
        """
        Saves the current plot.

        Parameters:
        - params: Model parameters.
        - plot_type: Type of plot being saved.
        - current_time: Current timestamp.
        """
        folder_name = f'images/{plot_type}'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        plt.savefig(f'{folder_name}/{plot_type}_{params}_{current_time}.png')
        plt.close()

    def save_data(self, train_rate, time_step, neuron, dropout_rate, optimizer, patience, epoch, batch_size, activation, kernel_regularizer, loss_function, history, test_loss, close_data, train_predict, test_predict, predictions, params):
        """
        Saves model data to the database.

        Parameters:
        - Various parameters related to model configuration and training.
        """
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Save model data to the database
        table_name = 'model_data'
        params = {
            'train_rate': train_rate,
            'time_step': time_step,
            'neuron': neuron,
            'dropout_rate': dropout_rate,
            'optimizer': optimizer,
            'patience': patience,
            'epoch': epoch,
            'batch_size': batch_size,
            'activation': activation,
            'kernel_regularizer': kernel_regularizer,
            'loss_function': loss_function,
        }
        training_data = {
            'epoch_used': epoch,
            'start_timestamp': int(history.epoch[0]),
            'finish_timestamp': int(history.epoch[-1]),
            'total_time': int(history.epoch[-1] - history.epoch[0]),
            'start_date': history.epoch[0],
            'finish_date': history.epoch[-1],
            'test_loss': test_loss,
            'training_loss': json.dumps(history.history['loss']),
            'validation_loss': json.dumps(history.history['val_loss']),
            'close_data': json.dumps([item for sublist in close_data.values.tolist() for item in sublist]),
            'train_predict': json.dumps(train_predict.tolist()),
            'test_predict': json.dumps(test_predict.tolist()),
            'predictions': json.dumps([item for sublist in predictions.tolist() for item in sublist])
        }

        self.db.save_data(params, training_data, table_name)

# Example usage:
# lstm_helper = LSTMModelHelper()
# lstm_helper.training_validation_loss_plot(history, params, current_time)
# lstm_helper.close_and_predictions_plot(df, close_data, train_date, train_predict, test_date, test_predict, params)
# lstm_helper.future_plot(future_dates, predictions, params, current_time)
# lstm_helper.save_data(train_rate, time_step, neuron, dropout_rate, optimizer, patience, epoch, batch_size, activation, kernel_regularizer, loss_function, test_loss, close_data, train_predict, test_predict, predictions, params)
