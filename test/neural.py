import numpy as np

# Input vector
X = np.array([200, 17])

# Weight matrix (2x3)
W = np.array([[1, -3, 5], 
              [-2, 4, -6]])
 
# Bias vector (for 3 neurons)
b = np.array([-1, 1, 2])

# Activation function (example: ReLU)
def g(z):
    return np.where(z > 0, 1, 0)  # Step function used in the example

# Implementing the dense layer using a for-loop
def dense_loop(a_in, W, b):
    units = W.shape[1]  # Number of output neurons
    a_out = np.zeros(units)  # Initialize output

    for j in range(units):
        w = W[:, j]  # Extract j-th column (weights for this neuron)
        z = np.dot(w, a_in) + b[j]  # Compute weighted sum + bias
        a_out[j] = g(z)  # Apply activation function

    return a_out

# Implementing the dense layer using vectorized NumPy operations
def dense_vectorized(a_in, W, b):
    z = np.dot(W.T, a_in) + b  # Compute weighted sum for all neurons at once
    return g(z)  # Apply activation function

# Compute output using both methods
output_loop = dense_loop(X, W, b)
output_vectorized = dense_vectorized(X, W, b)

# Print outputs
print("Output using for-loop:", output_loop)
print("Output using vectorized computation:", output_vectorized)