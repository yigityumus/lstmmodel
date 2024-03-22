import sqlite3
import json


class LSTMDatabase:
    def __init__(self):
        self.params_column_types = {
            'train_rate': 'REAL',
            'time_step': 'INTEGER',
            'neuron': 'INTEGER',
            'dropout_rate': 'REAL',
            'optimizer': 'TEXT',
            'patience': 'INTEGER',
            'epoch': 'INTEGER',
            'batch_size': 'INTEGER',
            'activation': 'TEXT',
            'kernel_regularizer': 'TEXT',
            'loss_function': 'TEXT',
        }
        self.other_column_types = {
            'test_loss': 'REAL',
            'close_data': 'TEXT',
            'train_predict': 'TEXT',
            'test_predict': 'TEXT',
            'predictions': 'TEXT'
        }


    def save_data(self, params, test_loss, close_data, train_predict, test_predict, predictions):
        """
        Saves model data to SQLite3 database.

        Parameters:
        - Various parameters related to model configuration and training.
        """

        # TODO - Add training and validation loss for the future models
        # # Serialize lists into JSON strings
        # training_loss_json = json.dumps(history.history['loss'])
        # validation_loss_json = json.dumps(history.history['val_loss'])

        close_data_json = json.dumps(close_data)
        train_predict_json = json.dumps(train_predict)
        test_predict_json = json.dumps(test_predict)
        predictions_json = json.dumps(predictions)

        # TODO - Create a database class

        # Connect to SQLite database
        conn = sqlite3.connect('model_data.db')
        c = conn.cursor()

        # FIXME - The code is way too complicated. Make it much more simpler. Hint: BaseModel or things like it.
        # Create a table to store model data if it doesn't exist
        c.execute('''CREATE TABLE IF NOT EXISTS model_data
             (id INTEGER PRIMARY KEY, {}, {})'''.format(', '.join([f'{key} {value}' for key, value in self.params_column_types.items()]), ', '.join([f'{key} {value}' for key, value in self.other_column_types.items()])))

        # FIXME - Move this code to the Database class once you create it.
        # Insert data into the database
        c.execute("INSERT INTO model_data (train_rate, time_step, neuron, dropout_rate, optimizer, patience, epoch, batch_size, activation, kernel_regularizer, loss_function, test_loss, close_data, train_predict, test_predict, predictions) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
          tuple(params[key] for key in self.params_column_types.keys()) + (test_loss, close_data_json, train_predict_json, test_predict_json, predictions_json))


        # Commit changes and close connection
        conn.commit()
        conn.close()
        



if __name__ == '__main__':
    # TODO - This should be in another file in it should be in loop within the folder.
    # JSON file
    f = open ("data_(0.75, 6, 25, 0.2, 'adam', 7, 50, 10, 'tanh', 0.01, 'mean_absolute_error').json", "r")
    
    # Reading from file
    data = json.loads(f.read())


    params = data['params']
    test_loss = data['test_loss']
    close_data = data['close_data']
    train_predict = data['train_predict']
    test_predict = data['test_predict']
    predictions = data['predictions']


    data2 = LSTMDatabase()
    data2.save_data(params=params, test_loss=test_loss, close_data=close_data, train_predict=train_predict, test_predict=test_predict, predictions=predictions)
