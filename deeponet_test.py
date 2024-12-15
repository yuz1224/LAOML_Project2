from matplotlib import pyplot as plt
from deeponet import DeepONet
import numpy as np


# Poisson 2D data generation
def generate_poisson_data(a_values, num_points=100):
    def analytical_solution(x, y, a):
        return a * np.sin(np.pi * x) * np.sin(np.pi * y)

    x = np.linspace(0, 1, num_points)
    y = np.linspace(0, 1, num_points)
    X, Y = np.meshgrid(x, y)

    inputs = []
    outputs = []
    for a in a_values:
        u = analytical_solution(X, Y, a)
        inputs.append(np.c_[np.full(X.ravel().shape, a), X.ravel(), Y.ravel()])
        outputs.append(u.ravel())

    return np.vstack(inputs).T, np.hstack(outputs)


# Data preparation
a_train = np.linspace(-1, 1, 5)  # Training values for a
a_test = np.linspace(-1, 1, 10)  # Testing unseen values for a
input_data, output_data = generate_poisson_data(a_train, num_points=50)

# Shuffle and split data
sample_num = input_data.shape[1]
permute_idx = np.random.permutation(sample_num)
input_data = input_data[:, permute_idx]
output_data = output_data[None, :][:, permute_idx]

split_ratio = 0.7
train_num = int(sample_num * split_ratio)
x_train, x_test = input_data[:, :train_num], input_data[:, train_num:]
y_train, y_test = output_data[:, :train_num], output_data[:, train_num:]

# DeepONet setup
branch_layer_sizes = [1, 32, 32, 80]  # Branch net: takes a_i as input
trunk_layer_sizes = [2, 32, 32, 80]  # Trunk net: takes (x, y) as input
deeponet = DeepONet(branch_layer_sizes, trunk_layer_sizes)

# Train DeepONet
record = deeponet.train(x_train[:1, :], x_train[1:, :], y_train, epochs=1000,
                        learning_rate=0.001, batch_size=64, opt_type="adam",
                        x_val=(x_test[:1, :], x_test[1:, :]), u_val=y_test)

# Test DeepONet
a_test_values = np.linspace(-1, 1, 10)  # Test on unseen a values
input_test, output_test = generate_poisson_data(a_test_values, num_points=50)
predicted_output = deeponet.infer(input_test[:1, :], input_test[1:, :])


# Visualization
def visualization_2D(x, y, u, title):
    plt.figure(figsize=(8, 6))
    plt.tricontourf(x, y, u, levels=100, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


visualization_2D(input_test[1, :], input_test[2, :], output_test, "Actual u(x, y)")
visualization_2D(input_test[1, :], input_test[2, :], predicted_output.ravel(), "Predicted u(x, y)")
visualization_2D(input_test[1, :], input_test[2, :], np.abs(predicted_output.ravel() - output_test), "Absolute Error")


# Training history
def plot_training_history(record):
    plt.figure(figsize=(8, 6))
    plt.plot(record["epoch"], record["train_loss"], label='Train Loss', marker='o')
    plt.plot(record["epoch"], record["test_loss"], label='Test Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_training_history(record)
