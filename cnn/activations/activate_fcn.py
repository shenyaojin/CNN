import numpy as np


def ReLU(z):
    """
    Args:
        z: The input value or array. It can be a scalar or a numpy array.

    Returns:
        The output of the Rectified Linear Unit (ReLU) activation function. For each element in the input array, the function calculates the maximum between 0 and the element itself.

    Example:

        >>> ReLU(3)
        3

        >>> ReLU(-2)
        0

        >>> ReLU(np.array([-1, 2, -3]))
        array([0, 2, 0])

    """
    return np.maximum(0, z)

def sigmoid(z):
    """
    Args:
        z: the input value

    Returns:
        The sigmoid of the input value
    """
    d = 1 + np.exp(-z)
    return 1. / d

def softmax(z):
    """
    Args:
        z: A numpy array representing the input values.

    Returns:
        A numpy array representing the output values after applying softmax function.

    Raises:
        None.

    Examples:
        z = np.array([[1, 2, 3], [4, 5, 6]])
        softmax(z)
        Output: array([[0.09003057, 0.24472847, 0.66524096],
                       [0.09003057, 0.24472847, 0.66524096]])
    """
    d = np.exp(z)
    return d / d.sum(axis = 1).reshape(-1, 1)

def tanh(z):
    b = np.exp(z)
    c = np.exp(-z)
    return (b - c) / (b + c)

def Linear(z):
    return z

def ReLU_gradient(h):
    return h > 0

def sigmoid_gradient(h):
    return np.multiply(h, (1 - h))

def softmax_gradient(h):
    return np.multiply(h, (1 - h))

def tanh_gradient(h):
    return 1 - np.power(h, 2)

def Linear_gradient(z):
    return 1