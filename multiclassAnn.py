import numpy as np
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
x, y = mnist["data"], np.array([float(i) for i in mnist["target"]])

x /= 255
digits = 10
examples = y.shape[0]

y.reshape(1, examples)

y_new = np.eye(digits)[y.astype('int32')]
y_new = y_new.T.reshape(digits, examples)

m = 60000
m_test = x.shape[0] - m

x_train, x_test = x[:m].T, x[m:].T
y_train, y_test = y_new[:,:m], y_new[:,m:]

def compute_multiclass_loss(Y, Y_hat):

    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1/m) * L_sum

    return L

def sigmoid(z):
    return 1/(1+np.exp(-z))

n_x = x_train.shape[0]
n_h = 64
learning_rate = 1

w1 = np.random.randn(n_h, n_x)
b1 = np.zeros((n_h, 1))
w2 = np.random.randn(digits, n_h)
b2 = np.zeros((digits, 1))

x = x_train
y = y_train

for i in range(2000):
        z1 = np.matmul(w1, x)
        a1 = sigmoid(z1)
        z2 = np.matmul(a2, w2)
        a2 = np.exp(z2)/np.sum(np.exp(z2), axis=0)
        
        cost = compute_multiclass_loss(y, a2)

        dz2 = a2 - y
        dw2 = (1./m) * np.matmul()