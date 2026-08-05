import numpy as np

# ========================================
#   NPZ FILE CONVERSION TO BIN
# ========================================

def read_params():

    data = np.load('../../input/mnist_network_parameters.npz')

    W1 = data["W1"]
    B1 = data["B1"]

    W2 = data["W2"]
    B2 = data["B2"]

    W3 = data["W3"]
    B3 = data["B3"]

    return W1, B1, W2, B2, W3, B3

def export_params(W1, B1, W2, B2, W3, B3):
    with open("../../input/w1.bin", "wb") as file:
        file.write(W1)
    with open("../../input/b1.bin", "wb") as file:
        file.write(B1)
    with open("../../input/w2.bin", "wb") as file:
        file.write(W2)
    with open("../../input/b2.bin", "wb") as file:
        file.write(B2)
    with open("../../input/w3.bin", "wb") as file:
        file.write(W3)
    with open("../../input/b3.bin", "wb") as file:
        file.write(B3)


# ========================================
#   RUNTIME PARAMETER SETTING
# ========================================

W1, B1, W2, B2, W3, B3 = read_params()

W1 = np.ascontiguousarray(W1, dtype=np.float32)
W2 = np.ascontiguousarray(W2, dtype=np.float32)
W3 = np.ascontiguousarray(W3, dtype=np.float32)
B1 = np.ascontiguousarray(B1, dtype=np.float32)
B2 = np.ascontiguousarray(B2, dtype=np.float32)
B3 = np.ascontiguousarray(B3, dtype=np.float32)

export_params(W1, B1, W2, B2, W3, B3)

# Confirming the data types and format
print(W1.flags['C_CONTIGUOUS'], W1.dtype, W1.shape)
print(W2.flags['C_CONTIGUOUS'], W2.dtype, W2.shape)
print(W3.flags['C_CONTIGUOUS'], W3.dtype, W3.shape)
print(B1.flags['C_CONTIGUOUS'], B1.dtype, B1.shape)
print(B2.flags['C_CONTIGUOUS'], B2.dtype, B2.shape)
print(B3.flags['C_CONTIGUOUS'], B3.dtype, B3.shape)