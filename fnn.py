from matplotlib import pyplot as plt
import numpy as np


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

class FeedforwardNeuralNetwork:
    def __init__(self, layer_sizes, activation_type = "relu"):
        """
        Initialize the feedforward neural network.
        
        Parameters:
        layer_sizes (list): List containing the number of neurons in each layer, including the layer size for input and output layers.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        if activation_type in ["relu", "sigmoid", "tanh"]:
            self.activation_type = activation_type
        else:
            raise ValueError(f"Invalid activation function '{self.activation_type}'. Supported types: relu, sigmoid, tanh.")
        self.weights = []
        self.biases = []
        
        # initialize all weights with N(0,1) * 0.01
        # Initialize all biases with 0
        for i in range(1, len(self.layer_sizes)):
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i - 1]) * 0.01)
            self.biases.append(np.zeros((self.layer_sizes[i], 1)))

    def activation(self, z):
        """
        Activation function.
        
        Parameters:
        z (numpy.ndarray): Input array.
        
        Returns:
        numpy.ndarray: Output array after applying the activation function.
        """
        
        # using relu function here
        if self.activation_type == "relu":
            return np.maximum(0, z)
        elif self.activation_type == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.activation_type == "tanh":
            return np.tanh(z)
        else:
            raise Exception("Invalid activation function")

    def activation_derivative(self, z):
        """
        Derivative of the activation function.
        
        Parameters:
        z (numpy.ndarray): Input array.
        
        Returns:
        numpy.ndarray: Output array after applying derivative of the activation function.
        """
        if self.activation_type == "relu":
            return (z > 0).astype(float)
        elif self.activation_type == "sigmoid":
            sigmoid = self.activation(z)
            return sigmoid * (1 - sigmoid)
        elif self.activation_type == "tanh":
            return 1 - self.activation(z)**2
        else:
            raise Exception("Invalid activation function")

    def feedforward(self, x):
        """
        Perform a feedforward pass through the network.
        
        Parameters:
        x (numpy.ndarray): Input array.
        
        Returns:
        numpy.ndarray: Output of the network.
        """
        a_values = [x] # values after activation functions
        z_values = [] # values before activation functions
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a_values[-1]) + b
            z_values.append(z)
            a = self.activation(z) if len(z_values) < self.num_layers - 1 else z
            a_values.append(a)
        return a_values, z_values

    def compute_cost(self, y_pred, y_train):
        """
        Compute the cost function.
        
        Parameters:
        y_pred (numpy.ndarray): Predicted labels.
        y_train (numpy.ndarray): True labels.
        
        Returns:
        float: Cost value.
        """
        # Using MSE Loss Function here
        return np.mean((y_train - y_pred)**2)
    
    def backpropagate(self, z_values, a_values, y_true):
        """
        Perform backpropagation to compute gradients.
        
        Parameters:
        z_values (list): outputs before activation functions in each layer.
        a_values (list): outputs after activation functions in each layer.
        y_true (numpy.ndarray): True labels.
        
        Returns:
        tuple: Gradients of weights and biases.
        """
        
        # initialize the gradient for a, w and b
        w_gradient = []
        b_gradient = []
        
        # no activation function for output layers
        # compute the loss for the output layer individually
        # let error be np.mean(y_pred - y)
        z_gradient = 2 * (a_values[-1] - y_true) / y_true.shape[1]  # shape = {1, batch_size}
        w_gradient.append(z_gradient @ a_values[-2].T) # shape = {1, a_{L-1}}
        b_gradient.append(np.sum(z_gradient, axis = -1)[:,None]) # shape = {1, 1}
        
        for l in range(2, self.num_layers):
            
            # compute the gradient of loss w.r.t z^{-l} first
            ## l start from the 2nd to last layer 
            ## we have dz^{-l} = W^{a_{-l+1}, a^{-l}}^T @ (derivative of activation * dz^{-l+1})
            
            ## the derivative of activation function
            # # derivative of a^{-l} w.r.t z^{-l} 
            act_de = self.activation_derivative(z_values[-l]) # shape = {a_{-l}, batch_size}
            
            ## the derivative of loss w.r.t z^{-l} 
            z_gradient = self.weights[-l + 1].T @ z_gradient * act_de # shape = {a_{-l}, batch_size}
            
            # the derivative of loss w.r.t w^{a^{-l}, a^{l-1}}
            w_gradient.append(z_gradient @ a_values[-l - 1].T) # shape = {a_{-l}, a_{-l - 1}}
            
            # the derivative of loss w.r.t b^{a^{-l}}
            b_gradient.append(np.sum(z_gradient, axis = -1)[:,None]) # shape = {a_{-l}, 1}
            
        return w_gradient[::-1], b_gradient[::-1]

    def update_parameters(self, nabla_w, nabla_b, learning_rate, opt_type = "sgd"):
        """
        Update the weights and biases using the computed gradients.
        
        Parameters:
        nabla_w (list): Gradients of weights.
        nabla_b (list): Gradients of biases.
        learning_rate (float): Learning rate.
        """
        if opt_type == "sgd":
            self.weights = [w - learning_rate * dw for w, dw in zip(self.weights, nabla_w)]
            self.biases = [b - learning_rate * db for b, db in zip(self.biases, nabla_b)]
        elif opt_type == "adam":
            if not hasattr(self, "adm_step"):
                self.adm_step = 1
                self.w_f = [np.zeros_like(w) for w in self.weights] # 1st order moment for weights 
                self.w_s = [np.zeros_like(w) for w in self.weights] # 2nd order moment for weights
                
                self.b_f = [np.zeros_like(b) for b in self.biases] # 1st order moment for biases
                self.b_s = [np.zeros_like(b) for b in self.biases] # 2nd order moment for biases
            
            # initial parameters
            eps = 1e-10
            beta1 = 0.9
            beta2 = 0.999
            factor1 = 1 - beta1**self.adm_step
            factor2 = 1 - beta2**self.adm_step
            
            for i in range(len(nabla_w)):
                self.b_f[i] = (beta1 * self.b_f[i] + (1-beta1) * nabla_b[i])
                self.w_f[i] = (beta1 * self.w_f[i] + (1-beta1) * nabla_w[i])
                
                self.b_s[i] = (beta2 * self.b_s[i] + (1-beta2) * nabla_b[i])
                self.w_s[i] = (beta2 * self.w_s[i] + (1-beta2) * nabla_w[i])
                
                b_f_hat = self.b_f[i]/factor1
                w_f_hat = self.w_f[i]/factor1
                
                b_s_hat = self.b_s[i]/factor2
                w_s_hat = self.w_s[i]/factor2
                
                self.weights[i] -= w_f_hat / (eps + np.sqrt(w_s_hat))
                self.biases[i] -= b_f_hat / (eps + np.sqrt(b_s_hat))
                
            self.adm_step +=1
        else:
            raise Exception(f"Invalid optimization type {opt_type}. Supported types: \"sgd\", \"adam\".")
        
    def train(self, x_train, y_train, epochs, learning_rate, batch_size, opt_type = "adm"):
        """
        Train the neural network using mini-batch gradient descent with early stopping.
        
        Parameters:
        x_train (numpy.ndarray): Training data.
        y_train (numpy.ndarray): Training labels.
        epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
        batch_size (int): Size of each mini-batch.
        """
        np.random.seed(42)
        num_samples = x_train.shape[1]
        for _ in range(epochs):
            idx = np.random.permutation(num_samples)
            x_train = x_train[:,idx]
            y_train = y_train[:,idx]

            # -( n // -d) is equivalent to ceil(n/d) in Python
            for i in range(-(num_samples // -batch_size)):
                if (i+1) * batch_size >= num_samples:
                    x_batch = x_train[:,i * batch_size:]
                    y_batch = y_train[:,i * batch_size:]
                else:
                    x_batch = x_train[:,i * batch_size: (i + 1) * batch_size]
                    y_batch = y_train[:,i * batch_size: (i + 1) * batch_size]
                
                # Forward & Backward Propagation
                a_values, z_values = self.feedforward(x_batch)
                nabla_w, nabla_b = self.backpropagate(z_values, a_values, y_batch)
                self.update_parameters(nabla_w, nabla_b, learning_rate, opt_type)

# In[] Initialize data
input_data, output_data = generate_data(num_points=50)
input_data = input_data.T
output_data = output_data[None,:]

# In[] testing code
nn = FeedforwardNeuralNetwork([2, 32, 32, 1], activation_type="relu")
x_train, x_test = input_data[:,:1500], input_data[:,1500:]
y_train, y_test = output_data[:,:1500], output_data[:,1500:]

# Train the network
nn.train(x_train, y_train, epochs=1000, learning_rate=0.1, batch_size=8, opt_type="sgd")

# Test error
activations, _ = nn.feedforward(x_train)
y_pred = activations[-1]
test_loss = nn.compute_cost(y_pred, y_train)

# Validate the network
activations, _ = nn.feedforward(x_test)
y_pred = activations[-1]
val_loss = nn.compute_cost(y_pred, y_test)




# In[]: the following for grid search

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