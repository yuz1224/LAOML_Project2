import numpy as np
import matplotlib.pyplot as plt

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