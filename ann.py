import numpy as np

# Load the MNIST Dataset from sklearn
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
images, labels = mnist["data"], np.array([float(i) for i in mnist["target"]])

images = images/255

def sigmoid(z):
    return 1/(np.exp(-z))

def computeLoss(y, y_hat):
    m = y.shape[1]
    l = -(1./m) * (np.sum(np.multiply(np.log(y_hat), y)) + np.sum(np.multiply(np.log(1-y_hat), (1-y))))

n_x = images.shape[0]
m = images.shape[1]

w = np.random.randn(n_x, 1) * 0.01
b = np.zeros((1, 1))


learning_rate = 1

print(np.matmul(w, images).shape)
"""
for i in range(2000):
    # Input - Output
    z = np.matmul(w.T, x) + b  # Multiply the weights and inputs
    a = sigmoid(z) # apply non-linearity to our multiplied value
"""