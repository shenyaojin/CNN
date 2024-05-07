import numpy as np


def cross_tropy(h, y):
    """
    Args:
        h: numpy array or pandas DataFrame
            The predicted values from a model.

        y: numpy array or pandas DataFrame
            The actual target values.

    Returns:
        cost: float
            The calculated cross entropy cost.
    """
    m = y.shape[0]
    
    # compute the cost
    J = np.multiply(-y, np.log(h)) - np.multiply((1 - y), np.log(1 - h))
    cost = np.sum(J) / m

    return cost

def MSE(h, y):
    """
    Args:
        h (np.ndarray): Predicted values.
        y (np.ndarray): Actual values.

    Returns:
        float: Mean squared error.

    """
    m = y.shape[0]
    
    # compute the cost
    J = np.power(h - y, 2)
    cost = np.sum(J)/2/m

    return cost