from fnn import FeedforwardNeuralNetwork
import numpy as np


class DeepONet:
    def __init__(self, branch_layer_sizes, trunk_layer_sizes):
        """
        Initialize the DeepONet architecture.

        Parameters:
        branch_layer_sizes (list): List containing the number of neurons in each layer for the branch net.
        trunk_layer_sizes (list): List containing the number of neurons in each layer for the trunk net.
        """
        self.branch_net = FeedforwardNeuralNetwork(branch_layer_sizes)
        self.trunk_net = FeedforwardNeuralNetwork(trunk_layer_sizes)

        # Ensure the output dimensions of the branch and trunk nets match
        assert branch_layer_sizes[-1] == trunk_layer_sizes[-1], "Output dimensions of branch and trunk nets must match"

    def feedforward(self, x_branch, x_trunk):
        """
        Perform a feedforward pass through the DeepONet f -> u.

        Parameters:
        x_branch (numpy.ndarray): Input array for the branch net {#dimension f, batch_size}.
        x_trunk (numpy.ndarray): Input array for the trunk net {#dimension u, query_points}.

        Returns:
        numpy.ndarray: Output of the DeepONet.
        """
        a_branch, z_branch = self.branch_net.feedforward(x_branch) # output shape = {D, batch_size}
        a_trunk, z_trunk = self.trunk_net.feedforward(x_trunk)   # output shape = {D, query_points}

        return a_branch, z_branch, a_trunk, z_trunk

    def infer(self, x_branch, x_trunk):
        a_branch, _, a_trunk, _ = self.feedforward(x_branch, x_trunk)
        return a_branch[-1].T @ a_trunk[-1]

    def compute_cost(self, y_branch, y_trunk, u_true):
        """
        Compute the cost function.

        Parameters:
        y_pred (numpy.ndarray): Predicted labels.
        y_true (numpy.ndarray): True labels. dimension must be {batch_size, query_points}.

        Returns:
        float: Cost value.
        """

        return 0.5 * np.mean((y_branch.T @ y_trunk - u_true) ** 2)

    def loss_gradient(self, y_branch, y_trunk, u_true):

        batch_size = y_branch.shape[1]
        query_points = y_trunk.shape[1]
        total_num = batch_size * query_points

        branch_grad = y_trunk @ (y_branch.T @ y_trunk - u_true).T / total_num        # shape = {D, batch_size}
        trunk_grad =  y_branch @ (y_branch.T @ y_trunk - u_true)  / total_num       # shape = {D, query_points}

        return branch_grad, trunk_grad


    def backpropagate(self, x_branch, x_trunk, u_true):
        """
        Perform backpropagation to compute gradients.

        Parameters:
        x_branch (numpy.ndarray): Input array for the branch net.
        x_trunk (numpy.ndarray): Input array for the trunk net.
        y (numpy.ndarray): True labels.

        Returns:
        tuple: Gradients of weights and biases for both branch and trunk nets.
        """

        a_branch, z_branch, a_trunk, z_trunk = self.feedforward(x_branch, x_trunk)
        branch_grad, trunk_grad = self.loss_gradient(a_branch[-1], a_trunk[-1], u_true)

        branch_w_grad, branch_b_grad = self.branch_net.backpropagate(z_branch, a_branch, branch_grad)
        trunk_w_grad, trunk_b_grad = self.trunk_net.backpropagate(z_trunk, a_trunk, trunk_grad)

        return branch_w_grad, branch_b_grad, trunk_w_grad, trunk_b_grad

    def update_parameters(self, nabla_w_branch, nabla_b_branch, nabla_w_trunk, nabla_b_trunk, learning_rate, opt_type = "sgd"):
        """
        Update the weights and biases using the computed gradients.

        Parameters:
        nabla_w_branch (list): Gradients of weights for the branch net.
        nabla_b_branch (list): Gradients of biases for the branch net.
        nabla_w_trunk (list): Gradients of weights for the trunk net.
        nabla_b_trunk (list): Gradients of biases for the trunk net.
        learning_rate (float): Learning rate.
        """

        self.branch_net.update_parameters(nabla_w_branch, nabla_b_branch, learning_rate, opt_type)
        self.trunk_net.update_parameters(nabla_w_trunk, nabla_b_trunk, learning_rate, opt_type)

    def train(self, x_branch, x_trunk, u_train, epochs, learning_rate, batch_size,
              opt_type = "sgd", x_val=None, u_val=None, interval = 10):

        record = None
        np.random.seed(42)
        function_num = x_branch.shape[1]
        query_points = x_trunk.shape[1]

        if x_val is not None and u_val is not None:
            val_branch, val_trunk = x_val
            record = {
                "epoch": [],
                "train_loss": [],
                "test_loss": []
            }

        for iter in range(epochs):
            branch_idx = np.random.permutation(function_num)
            trunk_idx = np.random.permutation(query_points)

            x_branch, x_trunk = x_branch[:, branch_idx], x_trunk[:, trunk_idx]

            # record the training and testing/validation loss during model training
            if iter % interval == 0 and record is not None:

                record["epoch"].append(iter + 1)

                y_branch, _, y_trunk, _ = self.feedforward(x_branch, x_trunk)
                train_loss = self.compute_cost(y_branch, y_trunk, u_train)
                record["train_loss"].append(train_loss)

                y_branch, _, y_trunk, _ = self.feedforward(val_branch, val_trunk)
                val_loss = self.compute_cost(y_branch, y_trunk, u_val)
                record["test_loss"].append(val_loss)

                if iter % (interval * 10) == 0:
                    print(f"(DeepONET) Epoch {iter + 1}: Train Loss = {train_loss:.6f}, Test Loss = {val_loss:.6f}")

            # Mini-batch training
            # -( n // -d) is equivalent to ceil(n/d) in Python
            for i in range(-(function_num // -batch_size)):
                if (i + 1) * batch_size >= function_num:
                    x_batch_branch = x_branch[:, i * batch_size:]
                    x_batch_trunk = x_trunk[:, i * batch_size:]
                    u_batch = u_train[:, i * batch_size:]
                else:
                    x_batch_branch = x_branch[:, i * batch_size: (i + 1) * batch_size]
                    x_batch_trunk = x_trunk[:, i * batch_size: (i + 1) * batch_size]
                    u_batch = u_train[:, i * batch_size: (i + 1) * batch_size]

                # Forward & Backward Propagation
                branch_w_grad, branch_b_grad, trunk_w_grad, trunk_b_grad = self.backpropagate(x_batch_branch, x_batch_trunk, u_batch)
                self.update_parameters(branch_w_grad, branch_b_grad, trunk_w_grad, trunk_b_grad, learning_rate, opt_type)
        return record


