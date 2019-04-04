import numpy as np

# Load the MNIST Dataset from sklearn
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
x, y = mnist["data"], [float(i) for i in mnist["target"]]

x = x/255

def sigmoid(z):
    return 1/(np.exp(-z))

def computeLoss(y, y_hat):
    m = y.shape[1]
    l = -(1./m) * (np.sum(np.multiply(np.log(y_hat), y)) + np.sum(np.multiply(np.log(1-y_hat), (1-y))))

n_x = x.shape[0]
m = x.shape[1]
n_hidden = 64

w = np.random.randn(n_x, n_hidden) * 0.01
b = np.zeros((1, 1))

w2 = np.random.randn(64) * 0.01

learning_rate = 1

for i in range(2000):
    # Input - Layer 2
    z = np.matmul(W.T, x) + b  # Multiply the weights and inputs
    a = sigmoid(z) # apply non-linearity to our multiplied values

    # Layer 2 - Output
    z2 = np.matmul()
