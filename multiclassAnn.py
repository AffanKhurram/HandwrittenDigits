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

x_train, x_test = x[:m], x[m:]
y_train, y_test = y_new[:,:m].T, y_new[:,m:].T

def compute_multiclass_loss(Y, Y_hat):

    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1/m) * L_sum

    return L

def sigmoid(z):
    return 1/(1+np.exp(-z))


n_x = x_train.shape[1]
n_h = 64
learning_rate = 1

w1 = np.random.randn(n_x, n_h)
b1 = np.zeros((1, n_h))
w2 = np.random.randn(n_h, digits)
b2 = np.zeros((1, digits))

x = x_train
y = y_train

for i in range(1):
        z1 = np.matmul(x, w1)
        a1 = sigmoid(z1)
        z2 = np.matmul(a1, w2)
        a2 = np.exp(z2)/np.sum(np.exp(z2), axis=0)

        dz2 = y - a2
        dw2 = (1./m) * np.matmul(dz2.T, a2)
        db2 = (1./m) * np.sum()