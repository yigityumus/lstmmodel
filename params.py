# Define lists for parameters
TRAIN_RATES = [0.75] # 0.7, 0.8
TIME_STEPS = [6, 24, 72, 168]
NEURONS = [25, 50, 100]
DROPOUT_RATES = [0.2] # 0.1, 0.3
OPTIMIZERS = ['adam', 'rmsprop']
PATIENCES = [7, 10] # 13
EPOCHS = [50, 100, 150]
BATCH_SIZES = [10, 25]
ACTIVATIONS = ['tanh'] # sigmoid
KERNEL_REGULARIZERS = [0.01]
LOSS_FUNCTIONS = ['mean_absolute_error', 'mean_squared_error'] # 'huber_loss', 'logcosh'


params = {
    'train_rates': TRAIN_RATES,
    'time_steps': TIME_STEPS,
    'neurons': NEURONS,
    'dropout_rates': DROPOUT_RATES,
    'optimizers': OPTIMIZERS,
    'patiences': PATIENCES,
    'epochs': EPOCHS,
    'batch_sizes': BATCH_SIZES,
    'activations': ACTIVATIONS,
    'kernel_regularizers': KERNEL_REGULARIZERS,
    'loss_functions': LOSS_FUNCTIONS
}

# TODO - Create a config file for the params.


if __name__ == '__main__':
    num_combinations = 1
    for param_values in params.values():
        num_combinations *= len(param_values)

    print("Number of different possible combinations:", num_combinations)