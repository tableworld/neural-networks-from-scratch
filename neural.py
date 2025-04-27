# neural.py
import numpy as np
import matplotlib.pyplot as plt

# Set a seed for reproducibility
np.random.seed(42)

# 1. Create toy data (XOR inputs and outputs)
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

outputs = np.array([
    [0],
    [1],
    [1],
    [0]
])

# 2. Initialize weights and biases
input_size = 2
hidden_size = 2
output_size = 1

# Random weights for input to hidden
weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.random.rand(hidden_size)

# Random weights for hidden to output
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_output = np.random.rand(output_size)

# Learning rate
learning_rate = 0.1

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

losses = []
# 3. Training loop
for epoch in range(10000):
    # ---- FORWARD PASS ----
    hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predictions = sigmoid(final_input)

    # ---- BACKPROPAGATION ----
    error = outputs - predictions
    d_predictions = error * sigmoid_derivative(predictions)

    error_hidden = d_predictions.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # ---- UPDATE WEIGHTS AND BIASES ----
    weights_hidden_output += hidden_output.T.dot(d_predictions) * learning_rate
    bias_output += np.sum(d_predictions, axis=0) * learning_rate

    weights_input_hidden += inputs.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0) * learning_rate

    # Print loss every 1000 epochs
    loss = np.mean(error**2)
    losses.append(loss)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
# Plot the loss over epochs
plt.plot(losses)
plt.title("Loss over Training")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
# 4. Final output after training
print("\nTraining complete!")
print("Final predictions:")
print(predictions.round(3))
