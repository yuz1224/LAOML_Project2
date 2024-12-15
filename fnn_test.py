from fnn import FeedforwardNeuralNetwork
from matplotlib import pyplot as plt
import numpy as np


# In[]: Data generation for the Poisson equation
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


# In[] Initialize data
input_data, output_data = generate_data(num_points=100)
sample_num = input_data.shape[0]
permute_idx = np.random.permutation(sample_num)
input_data = input_data.T[:,permute_idx]
output_data = output_data[None,:][:, permute_idx]

# In[] testing code
nn = FeedforwardNeuralNetwork([2, 32, 32, 1], activation_type="relu")
split_ratio = 0.7
train_num = int(sample_num * split_ratio)
x_train, x_test = input_data[:, :train_num], input_data[:, train_num:]
y_train, y_test = output_data[:, :train_num], output_data[:, train_num:]

# Train the network
# nn.train(x_train, y_train, epochs=1000, learning_rate=0.1, batch_size=64, opt_type="sgd") # 这个目前的效果还挺不错的, bs = 64
record = nn.train(x_train, y_train, epochs=1001, learning_rate=0.0001, batch_size=128, opt_type="adam", x_val=x_test, y_val=y_test)

# Test error
activations, _ = nn.feedforward(x_train)
y_pred = activations[-1]
test_loss = nn.compute_cost(y_pred, y_train)

# Validate the network
activations, _ = nn.feedforward(x_test)
y_pred = activations[-1]
val_loss = nn.compute_cost(y_pred, y_test)

print(test_loss, val_loss)
# TODO : 分析一下tanh, sigmoid, relu不work的原因

# In[] 可视化部分
def visualization_2D(x, y, u, title):
    plt.figure(figsize=(8, 6))
    plt.tricontourf(x, y, u, levels=100, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

visualization_2D(x_test[0,:], x_test[1,:], y_test[0,:], title= "Actual u(x, y)")
visualization_2D(x_test[0,:], x_test[1,:], y_pred[0,:], title="Predicted u(x, y)")

abs_error = np.abs(y_pred[0,:] - y_test[0,:])
visualization_2D(x_test[0,:], x_test[1,:], abs_error, title= "Abs. Error")

# In[] 误差时间历程曲线
def plot_training_history(record):
    """
    Plot the training and testing loss curves from the record dictionary.

    Parameters:
    record (dict): A dictionary containing "epoch", "train_loss", and "test_loss".
    """
    epochs = record.get("epoch", [])
    train_loss = record.get("train_loss", [])
    test_loss = record.get("test_loss", [])

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, test_loss, label='Test Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_training_history(record)

# In[]:
# TODO :the following for grid search 后面的都没有修改 by Yuhan WU
# TODO: early stopping is not applied here

# Split the data into folds manually for cross-validation
def split_folds(x, y, num_folds):
    """
    return the folds contains the list of (x_train, y_train, x_val, y_val)
    """
    fold_size = x.shape[1] // num_folds
    folds = []
    for i in range(num_folds):
        val_indices = list(range(i * fold_size, (i + 1) * fold_size))
        train_indices = list(set(range(x.shape[1])) - set(val_indices))
        x_train, x_val = x[:, train_indices], x[:, val_indices]
        y_train, y_val = y[:, train_indices], y[:, val_indices]
        folds.append((x_train, y_train, x_val, y_val))
    return folds

# structures for networks
architectures = [
    [2, 16, 16, 1],  # 2 hidden layers with 16 neurons each
    [2, 32, 32, 1],  # 2 hidden layers with 32 neurons each
    [2, 16, 16, 16, 1],  # 3 hidden layers with 16 neurons each
    [2, 8, 16, 16, 8, 1]
]

# parameters for cross validation
learning_rates = [0.0001, 0.001, 0.01, 0.1]
batch_sizes = [128, 64, 32, 16]
num_folds = 5

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