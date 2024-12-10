import numpy as np
import matplotlib.pyplot as plt

class FeedforwardNeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Initialize the feedforward neural network.

        Parameters:
        layer_sizes (list): List containing the number of neurons in each layer.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i - 1]) * 0.1 for i in range(1, self.num_layers)]
        self.biases = [np.zeros((layer_sizes[i], 1)) for i in range(1, self.num_layers)]

    def activation(self, z):
        """
        Activation function (ReLU).

        Parameters:
        z (numpy.ndarray): Input array.

        Returns:
        numpy.ndarray: Output array after applying the activation function.
        """
        return np.maximum(0, z)

    def activation_derivative(self, z):
        """
        Derivative of the activation function (ReLU).

        Parameters:
        z (numpy.ndarray): Input array.

        Returns:
        numpy.ndarray: Output array after applying derivative of the activation function.
        """
        return (z > 0).astype(float)

    def feedforward(self, x):
        """
        Perform a feedforward pass through the network.

        Parameters:
        x (numpy.ndarray): Input array.

        Returns:
        list: Activations at each layer.
        """
        activations = [x]
        z_values = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            z_values.append(z)
            a = self.activation(z) if len(activations) < self.num_layers else z  # No activation at the output layer
            activations.append(a)
        return activations, z_values

    def compute_cost(self, y_pred, y_true):
        """
        Compute the Mean Squared Error (MSE) cost function.

        Parameters:
        y_pred (numpy.ndarray): Predicted labels.
        y_true (numpy.ndarray): True labels.

        Returns:
        float: Cost value.
        """
        return np.mean((y_pred - y_true) ** 2)

    def backpropagate(self, activations, z_values, y_true):
        """
        Perform backpropagation to compute gradients.

        Parameters:
        activations (list): Activations at each layer.
        z_values (list): Z-values at each layer.
        y_true (numpy.ndarray): True labels.

        Returns:
        tuple: Gradients of weights and biases.
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # Compute the output layer error
        delta = (activations[-1] - y_true)  # MSE loss derivative
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)

        # Propagate backward through the layers
        for l in range(2, self.num_layers):
            z = z_values[-l]
            sp = self.activation_derivative(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)

        return nabla_w, nabla_b

    def update_parameters(self, nabla_w, nabla_b, learning_rate):
        """
        Update the weights and biases using the computed gradients.

        Parameters:
        nabla_w (list): Gradients of weights.
        nabla_b (list): Gradients of biases.
        learning_rate (float): Learning rate.
        """
        self.weights = [w - learning_rate * dw for w, dw in zip(self.weights, nabla_w)]
        self.biases = [b - learning_rate * db for b, db in zip(self.biases, nabla_b)]

    def train(self, x_train, y_train, epochs, learning_rate, batch_size):
        """
        Train the neural network using mini-batch gradient descent.

        Parameters:
        x_train (numpy.ndarray): Training data.
        y_train (numpy.ndarray): Training labels.
        epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
        batch_size (int): Size of each mini-batch.
        """
        m = x_train.shape[1]
        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            x_train = x_train[:, permutation]
            y_train = y_train[:, permutation]
            for k in range(0, m, batch_size):
                x_batch = x_train[:, k:k + batch_size]
                y_batch = y_train[:, k:k + batch_size]

                # Forward propagation
                activations, z_values = self.feedforward(x_batch)

                # Backward propagation
                nabla_w, nabla_b = self.backpropagate(activations, z_values, y_batch)

                # Update weights and biases
                self.update_parameters(nabla_w, nabla_b, learning_rate)

            # Calculate and print the cost after each epoch
            activations, _ = self.feedforward(x_train)
            cost = self.compute_cost(activations[-1], y_train)
            # print(f"Epoch {epoch + 1}, Cost: {cost:.6f}")

# Data generation for the Poisson equation
def generate_data(num_points=99):
    def analytical_solution(x, y):
        # Poisson equation analytical solution
        sol = np.sin(np.pi * x) * np.sin(np.pi * y)
        sol[(x == 1) | (y == 1)] = 0
        return sol
    x = np.linspace(0, 1, num_points)
    y = np.linspace(0, 1, num_points)
    X, Y = np.meshgrid(x, y)
    u = analytical_solution(X, Y)
    return np.c_[X.ravel(), Y.ravel()], u.ravel()

# Initialize data
input_data, output_data = generate_data(num_points=9)

# Define the grid search parameters
architectures = [
    [2, 16, 16, 1],  # 2 hidden layers with 16 neurons each
    [2, 32, 32, 1],  # 2 hidden layers with 32 neurons each
    [2, 16, 16, 16, 1]  # 3 hidden layers with 16 neurons each
]
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [8, 16, 32]
num_folds = 5


# Split the data into folds manually for cross-validation
def split_folds(x, y, num_folds):
    fold_size = x.shape[1] // num_folds
    folds = []
    for i in range(num_folds):
        val_indices = list(range(i * fold_size, (i + 1) * fold_size))
        train_indices = list(set(range(x.shape[1])) - set(val_indices))
        x_train, x_val = x[:, train_indices], x[:, val_indices]
        y_train, y_val = y[:, train_indices], y[:, val_indices]
        folds.append((x_train, y_train, x_val, y_val))
    return folds

# Initialize the data for training
x_data = input_data.T  # Transpose for (features, samples)
y_data = output_data.reshape(1, -1)  # Reshape for (outputs, samples)
folds = split_folds(x_data, y_data, num_folds)

# Define a function to train and evaluate the neural network for a single configuration
def evaluate_model(architecture, learning_rate, batch_size, folds):
    validation_losses = []
    for x_train, y_train, x_val, y_val in folds:
        # Initialize the network
        nn = FeedforwardNeuralNetwork(architecture)

        # Train the network
        nn.train(x_train, y_train, epochs=100, learning_rate=learning_rate, batch_size=batch_size)

        # Validate the network
        activations, _ = nn.feedforward(x_val)
        y_pred = activations[-1]
        val_loss = nn.compute_cost(y_pred, y_val)
        validation_losses.append(val_loss)

    return np.mean(validation_losses)


# Perform grid search
best_params = None
best_loss = float('inf')

for architecture in architectures:
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            avg_loss = evaluate_model(architecture, learning_rate, batch_size, folds)
            # print(f"Architecture: {architecture}, Learning Rate: {learning_rate}, Batch Size: {batch_size}, Validation Loss: {avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = (architecture, learning_rate, batch_size)

# Train the final model with the best parameters on the entire dataset
final_nn = FeedforwardNeuralNetwork(best_params[0])
final_nn.train(x_data, y_data, epochs=1000, learning_rate=best_params[1], batch_size=best_params[2])

# Print the best parameters
print(f"\nBest Parameters: Architecture: {best_params[0]}, Learning Rate: {best_params[1]}, Batch Size: {best_params[2]} with Loss: {best_loss:.6f}")

# Predict and visualize the results
activations, _ = final_nn.feedforward(x_data)
final_predictions = activations[-1].flatten()

# Plot the 2D heatmap
x = input_data[:, 0]
y = input_data[:, 1]
u = output_data

plt.figure(figsize=(8, 6))
plt.tricontourf(x, y, u, levels=100, cmap='viridis')
plt.colorbar(label="u(x, y)")
plt.title("Analytical solution of Poisson equation")
plt.xlabel("x")
plt.ylabel("y")

plt.figure(figsize=(8, 6))
plt.tricontourf(x, y, final_predictions, levels=100, cmap='viridis')
plt.colorbar(label='Predicted u(x, y)')
plt.title("Best FNN Approximation of Poisson Solution")
plt.xlabel('x')
plt.ylabel('y')
plt.show()