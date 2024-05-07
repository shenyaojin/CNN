import numpy as np


def MAE(y, h):
    """
    Calculates the Mean Absolute Error (MAE).

    Args:
        y: Target values.
        h: Predicted values.

    Returns:
        Tuple containing the MAE value and its formatted representation.

    Example:
        >>> y = np.array([1, 2, 3, 4])
        >>> h = np.array([2, 3, 4, 5])
        >>> MAE(y, h)
        ('MAE: 1.0      ', 1.0)
    """
    AE = np.abs(h - y)
    MAE = np.sum(AE.mean(axis = 0))
    return "MAE: %-10s" % np.round(MAE, 6), MAE


def categorical_accuracy(y, h):
    """
    Args:
        y: The true labels of the samples. This should be a 2-dimensional array with shape (n_samples, n_classes), where n_samples is the number of samples and n_classes is the number of
    * classes.
        h: The predicted labels of the samples. This should be a 2-dimensional array with shape (n_samples, n_classes), where n_samples is the number of samples and n_classes is the number
    * of classes.

    Returns:
        A tuple containing the accuracy as a formatted string and the accuracy as a floating-point number. The accuracy represents the proportion of samples for which the predicted label
    * matches the true label, and is calculated as the sum of the number of correct predictions divided by the total number of samples.

    Example usage:

        y_true = np.array([[0, 1, 0], [1, 0, 0]])
        y_pred = np.array([[0.1, 0.9, 0], [0.8, 0.1, 0.1]])
        acc_str, acc = categorical_accuracy(y_true, y_pred)
        print(acc_str, acc)
        # Output: accuracy: 0.5       0.5

    Note:
        The method assumes that both y and h have the same number of samples and classes.
    """
    accuracy = np.sum(y.argmax(axis=1) == h.argmax(axis=1)) / y.shape[0]
    return "accuracy: %-10s" % np.round(accuracy, 6), accuracy