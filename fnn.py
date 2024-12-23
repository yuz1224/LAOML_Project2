from matplotlib import pyplot as plt
import numpy as np

def visualization_2D(x, y, u, title):
    plt.figure(figsize=(8, 6))
    plt.tricontourf(x, y, u, levels=100, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

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
            raise ValueError(f"Invalid activation function '{activation_type}'. Supported types: relu, sigmoid, tanh.")
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
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s)
        elif self.activation_type == "tanh":
            t = np.tanh(z)
            return 1 - t**2
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

    def loss_gradient(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.shape[1]

    def backpropagate(self, z_values, a_values, loss_grad):
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
        z_gradient = loss_grad  # shape = {1, batch_size}
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
                self.b_f[i] = beta1 * self.b_f[i] + (1-beta1) * nabla_b[i]
                self.w_f[i] = beta1 * self.w_f[i] + (1-beta1) * nabla_w[i]
                
                self.b_s[i] = beta2 * self.b_s[i] + (1-beta2) * (nabla_b[i]**2)
                self.w_s[i] = beta2 * self.w_s[i] + (1-beta2) * (nabla_w[i]**2)
                
                b_f_hat = self.b_f[i]/factor1
                w_f_hat = self.w_f[i]/factor1
                
                b_s_hat = self.b_s[i]/factor2
                w_s_hat = self.w_s[i]/factor2
                
                self.weights[i] -= learning_rate * w_f_hat / (eps + np.sqrt(w_s_hat))
                self.biases[i] -= learning_rate * b_f_hat / (eps + np.sqrt(b_s_hat))
                
            self.adm_step +=1
        else:
            raise Exception(f"Invalid optimization type {opt_type}. Supported types: \"sgd\", \"adam\".")
        
    def train(self, x_train, y_train, epochs, learning_rate, batch_size, opt_type = "sgd", x_val=None, y_val=None, interval = 10):
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
        record = None

        if x_val is not None and y_val is not None:
            record = {
                "epoch": [],
                "train_loss": [],
                "test_loss": []
            }

        for iter in range(epochs):
            idx = np.random.permutation(num_samples)
            x_train = x_train[:,idx]
            y_train = y_train[:,idx]

            # record the training and testing/validation loss during model training
            if iter % interval == 0 and record is not None:
                a_values_full_train, _ = self.feedforward(x_train)
                train_loss = self.compute_cost(a_values_full_train[-1], y_train)

                record["epoch"].append(iter + 1)
                record["train_loss"].append(train_loss)

                a_values_full_val, _ = self.feedforward(x_val)
                val_loss = self.compute_cost(a_values_full_val[-1], y_val)
                record["test_loss"].append(val_loss)

                if iter % (interval * 10) == 0:
                    print(f"(FNN) Epoch {iter + 1}: Train Loss = {train_loss:.6f}, Test Loss = {val_loss:.6f}")

            # Mini-batch training
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
                loss_grad = self.loss_gradient(a_values[-1], y_batch)
                nabla_w, nabla_b = self.backpropagate(z_values, a_values, loss_grad)
                self.update_parameters(nabla_w, nabla_b, learning_rate, opt_type)
        return record