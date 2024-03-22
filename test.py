import sqlite3
import json

# TODO - Write some tests for the code.

# Connect to SQLite database
conn = sqlite3.connect('example.db')
c = conn.cursor()

# Create a table to store data
c.execute('''CREATE TABLE IF NOT EXISTS model_data
             (id INTEGER PRIMARY KEY, params TEXT, train_loss TEXT, val_loss TEXT)''')

# Sample data
params = {'param1': 10, 'param2': 20}
train_loss = [0.1, 0.2, 0.3, 0.4]
val_loss = [0.05, 0.1, 0.15, 0.2]

# Serialize lists into JSON strings
params_json = json.dumps(params)
train_loss_json = json.dumps(train_loss)
val_loss_json = json.dumps(val_loss)

# Insert data into the database
c.execute("INSERT INTO model_data (params, train_loss, val_loss) VALUES (?, ?, ?)", (params_json, train_loss_json, val_loss_json))
conn.commit()

# Retrieve data from the database
c.execute("SELECT * FROM model_data")
row = c.fetchone()

# Deserialize JSON strings back into Python lists and dictionary
retrieved_params = json.loads(row[1])
retrieved_train_loss = json.loads(row[2])
retrieved_val_loss = json.loads(row[3])

print("Retrieved params:", retrieved_params)
print("Retrieved train loss:", retrieved_train_loss)
print("Retrieved val loss:", retrieved_val_loss)

# Close the connection
conn.close()
