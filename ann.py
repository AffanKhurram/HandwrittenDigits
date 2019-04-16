import numpy as np

# Load the MNIST Dataset from sklearn
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
images, labels = mnist["data"], np.array([float(i) for i in mnist["target"]])

images = images/255

def sigmoid(z):
    return 1/(1+np.exp(-z))

def computeLoss(y, y_hat):
    m = y.shape[0]
    l = -(1./m) * (np.sum(np.multiply(np.log(y_hat), y)) + np.sum(np.multiply(np.log(1-y_hat), (1-y))))
    return l

# Redo the labels for the number 1
new_l = np.zeros(labels.shape)
new_l[np.where(labels == 1.0)[0]] = 1
labels = new_l

# Split the training and test datasets
m = 60000
m_test = images.shape[0] - m
images, images_test = images[:m], images[m:]
labels, labels_test = labels[:m], labels[m:]

# Variables for our model
n_x = images.shape[0] # Number of training sets
m = images.shape[1] # Size of each of the pictures
n_h = 64 # Number of nodes in our hidden layer

w1 = np.random.randn(m, n_h) # First set of weights (Input -> hidden)
b1 = np.zeros((n_h, 1)) # Bias for inputs

w2 = np.random.randn(n_h, 1) # Second set of weights (Hidden -> output)
b2 = np.zeros((1, 1)) # Bias for hidden

learning_rate = 1


for i in range(2000):
    # Input -> Output
    z1 = np.matmul(images, w1) # Multiply our inputs by the weights
    a1 = sigmoid(z1) # Apply non-linearity to our hidden layer nodes

    # Hidden -> Output
    z2 = np.matmul(a1, w2)
    a2 = sigmoid(z2)

    # Backpropagation
    # Output -> Hidden
    dz2 = a2 - labels.reshape(labels.shape[0], 1)       
    dw2 = (1./n_x) * np.matmul(a1.T, dz2)
    db2 = (1./n_x) * np.sum(dz2.T, axis=1, keepdims=True)
    # Hidden -> Input
    da1 = np.matmul(labels.reshape(labels.shape[0], 1), w2.T)
    
    dz1 = da1 * sigmoid(z1) * (1 - sigmoid(z1))
    dw1 = (1./n_x) * np.matmul(images.T, dz1)
    
    db1 = (1./n_x) * np.sum(dz1.T, axis=1, keepdims=True)

    # Change values
    w2 -= dw2 * learning_rate
    b2 -= db2 * learning_rate
    w1 -= dw1 * learning_rate
    b1 -= db1 * learning_rate

    if i % 100 == 0:
        print('Epoch ', i, 'cost ', computeLoss(labels.reshape(labels.shape[0], 1), a2))


    