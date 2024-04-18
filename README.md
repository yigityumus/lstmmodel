# LSTM Model Helper

This repository contains Python scripts for training LSTM models, saving data to a SQLite database, and visualizing the results.

## Prerequisites

- Python 3.9 or higher
- [DB Browser for SQLite](https://sqlitebrowser.org/) (version X.X.X)

## Files

- `lstm_model_helper.py`: Contains the `LSTMModelHelper` class with methods for creating folders, plotting training/validation loss, plotting close values and predictions, plotting future predictions, and saving images.
- `lstm_database.py`: Contains the `LSTMDatabase` class with methods for creating a SQLite database, creating tables, and inserting data into the database.
- `main.py`: Script for training LSTM models with different parameter combinations, plotting results, and saving data to the database.

## Installation using Docker

1. Make sure you have Docker installed on your machine.
2. Clone this repository to your local machine.
3. Navigate to the root directory of the repository.
4. Build the Docker image using the following command:
   ```bash
   docker build -t lstm-model .
   ```
5. Run the Docker container:
   ```bash
   docker run lstm-model
   ```

## Installation without Docker

### For macOS and Linux:

1. Clone this repository to your local machine.
2. Open Terminal and navigate to the root directory of the repository.
3. Create a Python virtual environment named `mlp_lstm`:

   ```bash
   python3 -m venv mlp_lstm
   ```

4. Activate the virtual environment:
   ```bash
   source mlp_lstm/bin/activate
   ```
5. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
6. Run the `main.py` script:
   ```bash
   python main.py
   ```
7. After you're done, deactivate the virtual environment:
   ```bash
   deactivate
   ```

### For Windows:

1. Clone this repository to your local machine.
2. Open Command Prompt and navigate to the root directory of the repository.
3. Create a Python virtual environment named `mlp_lstm`:
   ```bash
   python -m venv mlp_lstm
   ```
4. Activate the virtual environment:
   ```bash
   .\mlp_lstm\Scripts\activate
   ```
5. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
6. Run the `main.py` script:
   ```bash
   python main.py
   ```
7. After you're done, deactivate the virtual environment:
   ```bash
   deactivate
   ```
