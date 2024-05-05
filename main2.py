from model import Model
import tensorflow as tf
import os
import sys
import datetime as dt



if __name__ == '__main__':
    current_time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Current time: {current_time}")

    # model_type = sys.argv[1].upper()
    

    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Set TensorFlow logging level to suppress informational messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
    # 0: all messages, 1: INFO messages, 2: INFO and WARNING messages, 3: INFO, WARNING, and ERROR messages

    # model = Model(model_type, 'new_params.json')
    # model.run()

    # lstm_model = Model('LSTM', 'new_params.json')
    # lstm_model.run()

    bilstm_model = Model('BILSTM', 'new_params.json')
    bilstm_model.run()
