import sqlite3

class LSTMDatabase:
    def __init__(self, database_name: str):
        """
        Initialize LSTMDatabase object.

        Parameters:
        - database_name: Name of the SQLite database file.
        """
        try:
            # TODO - Search if using pydantic models would be better. Find a way out from specifying each param as TEXT.
            self.db_name = database_name
            self.conn = sqlite3.connect(self.db_name)
            self.params = {
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
            self.training_data = {
                'epoch_used': 'INTEGER',
                'start_timestamp': 'INTEGER',
                'finish_timestamp': 'INTEGER',
                'total_time': 'INTEGER',
                'start_date': 'TEXT',
                'finish_date': 'TEXT',
                'test_loss': 'REAL',
                'training_loss': 'TEXT',
                'validation_loss': 'TEXT',
                'close_data': 'TEXT',
                'train_predict': 'TEXT',
                'test_predict': 'TEXT',
                'predictions': 'TEXT'
            }
        except Exception as e:
            print(f"Error occurred during initialization: {str(e)}")


    def create_table(self, table_name: str) -> None:
        """
        Create a table in the database.

        Parameters:
        - table_name: Name of the table to be created.
        """
        try:
            # Connect to SQLite database
            cur = self.conn.cursor()

            # Get the keys and data types from params and training_data.
            params_items = [f'{key} {value}' for key, value in self.params.items()]
            training_items = [f'{key} {value}' for key, value in self.training_data.items()]

            # Create a table
            query = '''CREATE TABLE IF NOT EXISTS {}
                (id INTEGER PRIMARY KEY, {}, {})
                '''.format(table_name, ', '.join(params_items), ', '.join(training_items))
            cur.execute(query)
            print(f"Table {table_name} created successfully.")

            # Commit changes and close connection
            self.conn.commit()
            self.conn.close()
        except Exception as e:
            print(f"Error occurred during table creation: {str(e)}")


    def insert_data(self, params: dict, training_data: dict, table_name: str) -> None:
        """
        Insert data into the table.

        Parameters:
        - params: Dictionary containing parameter values.
        - training_data: Dictionary containing training data values.
        - table_name: Name of the table to insert data into.
        """
        try:
            # Connect to SQLite database
            cur = self.conn.cursor()

            # Prepare column names and placeholders for the INSERT statement
            columns = ', '.join(self.params.keys()) + ', ' + ', '.join(self.training_data.keys())
            placeholders = ', '.join(['?'] * (len(self.params) + len(self.training_data)))

            # Prepare values for the INSERT statement
            values = tuple(params.get(key, '') for key in self.params.keys()) + \
                     tuple(training_data.get(key, '') for key in self.training_data.keys())

            # Execute the INSERT statement
            query = "INSERT INTO {} ({}) VALUES ({})".format(table_name, columns, placeholders)
            cur.execute(query, values)

            # Commit changes and close connection
            self.conn.commit()
            self.conn.close()
        except Exception as e:
            print(f"Error occurred during data insertion: {str(e)}")


    def save_data(self, params: dict, training_data: dict, table_name: str) -> None:
        """
        Create table (if not exists) and insert data into it.

        Parameters:
        - params: Dictionary containing parameter values.
        - training_data: Dictionary containing training data values.
        - table_name: Name of the table to insert data into.
        """
        try:
            self.create_table(table_name)
            self.insert_data(params, training_data, table_name)
        except Exception as e:
            print(f"Error occurred during save data process: {str(e)}")
