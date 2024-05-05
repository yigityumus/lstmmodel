import sqlite3
import csv
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np


from functions import load_params_from_json




actual_data_file = 'csv_files/ETHUSDT_1h_afterwards.csv'
predictions_file = 'residual_calculations.csv'


def get_min_test_losses(db_name, table_name, options):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Convert tuple of column names to a comma-separated string
    options_str = ', '.join(options)

    query = f"SELECT {options_str} FROM {table_name} ORDER BY test_loss ASC"
    cursor.execute(query)

    min_test_losses = cursor.fetchall()
    conn.close()
    return min_test_losses



def save_list_of_tuples_to_csv(data, headers, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in data:
            writer.writerow(row)

    print(f"CSV file {filename} created successfully.")



params = load_params_from_json('new_params.json')
params_keys = [key for key in params['parameter_list'].keys()]

other_keys = ['id', 'test_loss', 'predictions']

index_to_insert = 2

# Merge the lists
select_options = other_keys[:index_to_insert] + params_keys + other_keys[index_to_insert:]
print(select_options)

model_type = 'BILSTM'
table_name = f"ETHUSDT_1h_FUTURES_{model_type}"
table_name = f"{params['data']['security']}_{params['data']['interval']}_{params['data']['data_type']}_{model_type}"
print(table_name)


min_losses = get_min_test_losses(params['database_name'], table_name, select_options)
print("min_losses fetched successfully.")
print(min_losses)


save_list_of_tuples_to_csv(min_losses, select_options, predictions_file)





def parse_predictions(predictions_subset):
    # Convert the string representation of list to a Python list
    predictions_subset_list = ast.literal_eval(predictions_subset)
    
    # Ensure each element is a float
    predictions_subset_float = [float(value) for value in predictions_subset_list]
    return predictions_subset_float


def calculate_residuals(actual_data_file, predictions_file):
    my_residuals = []
    actual_data_df = pd.read_csv(actual_data_file)

    predictions_df = pd.read_csv(predictions_file)
    
    for index, row in predictions_df.iterrows():
        id = row['id']
        # Get the 'time_step' value from the current row
        time_step = row['time_step']

        # Fetch the first 'time_step' number of data points from the actual data
        actual_data_subset = actual_data_df['close'].head(time_step).tolist()

        # Get the corresponding predictions and convert to a list
        predictions_subset = row['predictions']
        predictions_subset_float = parse_predictions(predictions_subset)

        # Calculate residuals by subtracting actual data from predictions
        residuals = [pred - actual for pred, actual in zip(predictions_subset_float, actual_data_subset)]
        residuals_percentage = [calc_percentage(pred, actual) for pred, actual in zip(predictions_subset_float, actual_data_subset)]
        sum_residuals = sum(residuals)
        sum_abs_residuals = sum(abs(residual) for residual in residuals)
        sum_residuals_percentage = sum(residuals_percentage)
        sum_abs_residuals_percentage = sum(abs(residual_perc) for residual_perc in residuals_percentage)

        data = {
            'id': id,
            'time_step': time_step,
            'residuals': residuals,
            'sum_residuals': sum_residuals,
            'sum_abs_residuals': sum_abs_residuals,
            'residuals_percentage': residuals_percentage,
            'sum_residuals_percentage': sum_residuals_percentage,
            'sum_abs_residuals_percentage': sum_abs_residuals_percentage
        }
        
        my_residuals.append(data)

    df = pd.DataFrame(my_residuals)
    return df


def calc_percentage(pred, actual):
    return float((pred - actual) / actual * 100)



residuals = calculate_residuals(actual_data_file, predictions_file)
print(residuals.head())



# Save the DataFrame as a CSV file
residuals_data_path = f'residuals/{model_type}/residuals_options.csv'
residuals.to_csv(residuals_data_path, index=False)
print("DataFrame saved as CSV successfully.")



df1 = pd.read_csv(predictions_file)
df2 = pd.read_csv(residuals_data_path)

# Add 'residuals', 'sum_residuals', and 'sum_abs_residuals' columns from df2 to df1
df1['residuals'] = df2['residuals']
df1['sum_residuals'] = df2['sum_residuals']
df1['sum_abs_residuals'] = df2['sum_abs_residuals']
df1['residuals_percentage'] = df2['residuals_percentage']
df1['sum_residuals_percentage'] = df2['sum_residuals_percentage']
df1['sum_abs_residuals_percentage'] = df2['sum_abs_residuals_percentage']

# Save the modified DataFrame to a new CSV file
output_file = f'residuals/{model_type}/predictions_residuals.csv'
df1.to_csv(output_file, index=False)



def plot_residuals(model_type, predictions, residuals, residuals_percentage, file_naming_params):
    # Convert text representation of lists to actual lists
    predictions = ast.literal_eval(predictions)
    residuals = ast.literal_eval(residuals)
    residuals_percentage = ast.literal_eval(residuals_percentage)
    
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

    plt.title(f'Residuals vs. Predictions {file_naming_params} Plot - {model_type}')

    # Save the plot
    output_filename = f'/Users/yigityumus/Documents/GSU/Cours/Semestre_8/BitirmeTezi/residuals/{model_type}/residuals_{file_naming_params}.png'
    plt.savefig(output_filename)
    plt.close()






def create_residual_plots(df: pd.DataFrame, model_type):
    # Iterate over rows and create graphs
    for index, row in df.iterrows():
        columns_to_extract = ['id', 'time_step', 'neuron', 'optimizer', 'patience', 'epoch', 'batch_size', 'activation', 'loss_function']
        file_naming_params = tuple(row[columns_to_extract].values.tolist())
        # Call plot_residuals function
        plot_residuals(model_type, row['predictions'], row['residuals'], row['residuals_percentage'], file_naming_params)
    print("Plots saved successfully.")



df = pd.read_csv(f'residuals/{model_type}/predictions_residuals.csv')
create_residual_plots(df, model_type)
