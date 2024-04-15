import os
import json
import matplotlib.pyplot as plt


# TODO - Remove this file once you complete the other TODOs


class LSTMModelHelper:
    def __init__(self):
        pass

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
        Saves model data to JSON file.

        Parameters:
        - Various parameters related to model configuration and training.
        """
        # Prepare JSON object
        data = {
            'params': {
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
            },
            # 'training_loss': history.history['loss'],
            # 'validation_loss': history.history['val_loss'],
            'test_loss': test_loss,
            'close_data': [item for sublist in close_data.values.tolist() for item in sublist],
            'train_predict': train_predict.tolist(),
            'test_predict': test_predict.tolist(),
            'predictions': [item for sublist in predictions.tolist() for item in sublist]
        }

        data_folder_name = 'model_data_files'
        if not os.path.exists(data_folder_name):
            os.makedirs(data_folder_name)

        # Save as JSON
        with open(f'{data_folder_name}/data_{params}.json', 'w') as f:
            json.dump(data, f)

# Example usage:
# lstm_helper = LSTMModelHelper()
# lstm_helper.training_validation_loss_plot(history, params, current_time)
# lstm_helper.close_and_predictions_plot(df, close_data, train_date, train_predict, test_date, test_predict, params)
# lstm_helper.future_plot(future_dates, predictions, params, current_time)
# lstm_helper.save_data(train_rate, time_step, neuron, dropout_rate, optimizer, patience, epoch, batch_size, activation, kernel_regularizer, loss_function, test_loss, close_data, train_predict, test_predict, predictions, params)
