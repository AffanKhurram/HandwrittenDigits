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

w = np.random.randn(n_x, 1) * 0.01
b = np.zeros((1, 1))

