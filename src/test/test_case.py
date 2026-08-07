import numpy as np

data = np.load('../../input/test_case.npz')

X = data["test_case"]

X = X.flatten()

with open("../../input/test_case.bin", "wb") as file:
        file.write(X)

def col_operation(A, W, B):
    X = (W @ A) + B

    Y = 1 / (1 + np.exp(-X))

    return Y

def forward_prop(X, W1, B1, W2, B2, W3, B3):
    A1 = col_operation(X, W1, B1)
    A2 = col_operation(A1, W2, B2)
    A3 = col_operation(A2, W3, B3)

    return A3

def read_params():

    data = np.load('../../input/mnist_network_parameters.npz')

    W1 = data["W1"]
    W2 = data["W2"]
    W3 = data["W3"]

    B1 = data["B1"].flatten()
    B2 = data["B2"].flatten()
    B3 = data["B3"].flatten()

    return W1, B1, W2, B2, W3, B3

W1, B1, W2, B2, W3, B3 = read_params()

for i in range(2):
    Y = forward_prop(X[(i * 784):(i * 784) + 784], W1, B1, W2, B2, W3, B3)

    print(Y)   