import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

def generate_random_array(size=100, seed=None):
    """
    Generate an array with random numbers.

    Parameters:
    - size (int): The size of the array (default is 100).
    - seed (int or None): Optional seed for random number generation.

    Returns:
    - numpy.ndarray: Array containing random numbers.
    """
    if seed is not None:
        np.random.seed(seed)
    random_array = np.random.uniform(-100, 100, size)
    return random_array

# Define the target function
def target_function(x):
    return x * np.sin(x**2 / 300)

# Define the interval for x
x_min = -100
x_max = 100

# Define the number of data points
num_points = 1000

# Generate evenly spaced values for x
np.random.seed(42)
x_train = np.linspace(x_min, x_max, num_points)

x_train, x_test = train_test_split(x_train, test_size=0.6, random_state=None)
print(x_train.shape)
print(x_test.shape)
y_train = target_function(x_train)
y_test = target_function(x_test)

# Define model architectures
architectures = [
    [50],               # Model 1 One hidden layer with 50 neurons
    [30, 20],           # Model 2 Two hidden layers with 30 and 20 neurons
    [20, 20, 20]        # Model 3 Three hidden layers with 20 neurons each
]

# Train and evaluate each model
for i, architecture in enumerate(architectures):
    print(f"Training model {i+1} with architecture: {architecture}")
    
    # Build the model
    model = keras.Sequential()
    model.add(keras.layers.Dense(architecture[0], activation='relu', input_shape=(1,)))
    for units in architecture[1:]:
        model.add(keras.layers.Dense(units, activation='relu'))
    model.add(keras.layers.Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, epochs=75, batch_size=16, verbose=0)

    # Evaluate the model on the training data
    train_loss = model.evaluate(x_train, y_train, verbose=0)
    print(f"Training loss for model {i+1}: {train_loss}")

    # Evaluate the model on the test data
    test_loss = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss for model {i+1}: {test_loss}")

    x_data = np.linspace(x_min, x_max, 1000)
    y_data = target_function(x_data)
    # Plot the true function
    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data, label='True Function', color='blue')

    # Plot the model's predictions
    plt.plot(x_test, model.predict(x_test), label=f'Model {i+1} Prediction', color='orange')

    plt.title(f"Model {i+1} with architecture: {architecture}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()