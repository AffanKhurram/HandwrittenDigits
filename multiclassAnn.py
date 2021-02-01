
import numpy as np
def compute_multiclass_loss(Y, Y_hat):
        Y = Y.T
        Y_hat = Y_hat.T
        L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
        m = Y.shape[1]
        L = -(1/m) * L_sum

        return L

def sigmoid(z):
        return 1/(1+np.exp(-z))


def forward_pass (img):
        w1 = np.loadtxt('W1.txt')
        b1 = np.loadtxt('b1.txt')
        w2 = np.loadtxt('W2.txt')
        b2 = np.loadtxt('b2.txt')
        img = img.reshape(784)
        z1 = np.matmul(img, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.matmul(a1, w2) + b2
        a2 = np.exp(z2.T)/np.sum(np.exp(z2.T), axis=0)
        a2 = a2.T

        print(a2)
        print(np.argmax(a2, axis=0))




if __name__ == '__main__':
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


        n_x = x_train.shape[1]
        n_h = 64
        learning_rate = 1

        w1 = np.loadtxt('W1.txt')
        b1 = np.loadtxt('b1.txt')
        w2 = np.loadtxt('W2.txt')
        b2 = np.loadtxt('b2.txt')

        x = x_train
        y = y_train

        z1 = np.matmul(x_test, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.matmul(a1, w2) + b2
        a2 = np.exp(z2.T)/np.sum(np.exp(z2.T), axis=0)
        a2 = a2.T       

        predictions = np.argmax(a2, axis=1)
        labels = np.argmax(y_test, axis=1)

        from sklearn.metrics import classification_report, confusion_matrix

        print(confusion_matrix(predictions, labels))
        print(classification_report(predictions, labels))

for i in range(0):
        z1 = np.matmul(x, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.matmul(a1, w2) + b2
        a2 = np.exp(z2.T)/np.sum(np.exp(z2.T), axis=0)
        a2 = a2.T

        dz2 = a2-y
        dw2 = (1./m) * np.matmul(a1.T, dz2)
        db2 = (1./m) * np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.matmul(dz2, w2.T)
        dz1 = da1 * sigmoid(z1) * (1 - sigmoid(z1))
        dw1 = (1./m) * np.matmul(x.T, dz1)
        db1 = (1./m) * np.sum(dz1, axis=0, keepdims=True)


        w2 -= learning_rate*dw2
        w1 -= learning_rate*dw1
        b2 -= learning_rate*db2
        b1 -= learning_rate*db1

        if (i%100 == 0):
                print('Epoch ', i, ' cost ', compute_multiclass_loss(y, a2))





