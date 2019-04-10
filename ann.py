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

n_x = images.shape[0] # Number of training sets
m = images.shape[1] # Size of each of the pictures

w = np.random.randn(m, 1) * 0.01
b = np.zeros((1, 1))


learning_rate = 1


for i in range(2000):
    # Input -> Output
    z = np.matmul(images, w) + b  # Multiply the weights and inputs
    a = sigmoid(z) # apply non-linearity to our multiplied value

    dw = (1/n_x) * np.matmul(images.T, (a-labels.reshape(labels.shape[0], 1)))
    db = (1/n_x) * np.sum((a.T-labels), axis=1, keepdims=True)

    w -= learning_rate * dw
    b -= learning_rate * db

    if i % 100 == 0:
        print('Epoch ', i,' cost: ', computeLoss(labels, a.T))

    
from sklearn.metrics import classification_report, confusion_matrix

z = np.matmul(images_test, w) + b
a = sigmoid(z)

predictions = (a > 0.5)[0:]
ls = (labels_test == 1)[0:]

print(confusion_matrix(predictions, ls))