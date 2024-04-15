import sqlite3
import json
from pydantic import BaseModel, Field
from typing import List


class ModelParams(BaseModel):
    train_rate: float
    time_step: int
    neuron: int
    dropout_rate: float
    optimizer: str
    patience: int
    epoch: int
    batch_size: int
    activation: str
    kernel_regularizer: float
    loss_function: str


class TrainingData(BaseModel):
    epoch_used: int
    start_timestamp: int
    start_date: str
    finish_timestamp: int
    finish_date: str
    total_time: int

    test_loss: List[float]
    training_loss: List[float]
    validation_loss: List[float]
    close_data: List[float]
    train_predict: List[float]
    test_predict: List[float]
    predictions: List[float]


class LSTMDatabase:
    def create_table_if_not_exists(self, db_name: str, table_name: str, other_column_types: TrainingData):
        """
        Creates a table to store model data if it doesn't exist.

        Parameters:
        - table_name: Name of the table to create
        - other_column_types: OtherColumnTypes object containing column types for non-parametric data
        """
        conn = sqlite3.connect(db_name)
        cur = conn.cursor()
        
        query = '''CREATE TABLE IF NOT EXISTS {}
            (id INTEGER PRIMARY KEY, {}, {})'''.format(table_name, ', '.join([f'{key} {value}' for key, value in ModelParams.__annotations__.items()]), ', '.join([f'{key} {value}' for key, value in other_column_types.dict().items()]))
        
        cur.execute(query)
        
        conn.commit()
        conn.close()


    def insert_data(self, db_name: str, table_name: str, params: ModelParams, other_column_types: TrainingData, close_data, train_predict, test_predict, predictions):
        """
        Inserts model data into the database.

        Parameters:
        - table_name: Name of the table to insert data into
        - params: ModelParams object containing various parameters related to model configuration and training
        - other_column_types: OtherColumnTypes object containing column types for non-parametric data
        """
        close_data_json = json.dumps(close_data)
        train_predict_json = json.dumps(train_predict)
        test_predict_json = json.dumps(test_predict)
        predictions_json = json.dumps(predictions)

        conn = sqlite3.connect(db_name)
        cur = conn.cursor()
        cur.execute(f"INSERT INTO {table_name} ({', '.join(params.dict().keys())}, {', '.join(other_column_types.dict().keys())}) VALUES ({', '.join(['?' for _ in params.dict().keys()])}, {', '.join(['?' for _ in other_column_types.dict().keys()])})",
            tuple(params.dict().values()) + tuple(other_column_types.dict().values()))
        conn.commit()
        conn.close()


    def save_data(self, params: ModelParams, other_column_types: TrainingData, close_data, train_predict, test_predict, predictions, table_name):
        """
        Saves model data to SQLite3 database.

        Parameters:
        - params: ModelParams object containing various parameters related to model configuration and training
        - other_column_types: OtherColumnTypes object containing column types for non-parametric data
        - table_name: Name of the table to save data into
        """
        self.create_table_if_not_exists(table_name, other_column_types)
        self.insert_data(table_name, params, other_column_types, close_data, train_predict, test_predict, predictions)


if __name__ == '__main__':
    # JSON file
    f = open("data_(0.75, 6, 25, 0.2, 'adam', 7, 50, 10, 'tanh', 0.01, 'mean_absolute_error').json", "r")
    
    # Reading from file
    data = json.loads(f.read())

    params_data = data['params']
    test_loss = data['test_loss']
    close_data = data['close_data']
    train_predict = data['train_predict']
    test_predict = data['test_predict']
    predictions = data['predictions']

    # Create ModelParams object
    params = ModelParams(**params_data)

    # Create OtherColumnTypes object
    other_column_types = TrainingData(**{key: [float(val) for val in data[key]] for key in data if key != 'params'})

    data2 = LSTMDatabase()
    table_name = 'LSTM_model'  # You can replace this with any desired table name
    data2.save_data(params=params, other_column_types=other_column_types, close_data=close_data, train_predict=train_predict, test_predict=test_predict, predictions=predictions, table_name=table_name)
