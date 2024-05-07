import numpy as np
import math


def Kaiming_uniform(fan_in, fan_out, a = math.sqrt(5)):
    """
    Args:
        fan_in (int): The number of input units.
        fan_out (int): The number of output units.
        a (float, optional): The negative slope of the rectifier used in the non-linear activation function. Default is math.sqrt(5).

    Returns:
        ndarray: A numpy.ndarray object representing the weight matrix with a shape of (fan_out, fan_in). The values are randomly generated from a uniform distribution between -bound and
    * bound.

    Note:
        This method implements the Kaiming uniform initialization, an initializer for weights in a neural network, introduced in the paper "Delving Deep into Rectifiers: Surpassing Human
    *-Level Performance on ImageNet Classification" by He, K., Zhang, X., Ren, S., Sun, J. (2015).
        The Kaiming uniform initialization scales the weights according to the number of input and output units, ensuring that the variance of the outputs remains approximately the same
    * as the variance of the inputs, which helps in improving the performance of deep neural networks.

    Example Usage:
        >>> Kaiming_uniform(256, 512, a=0.01)
    """
    bound = 6.0 / (1 + a * a) / fan_in
    bound = math.sqrt(bound)
    return np.random.uniform(low=-bound, high=bound, size=(fan_out, fan_in))


def Kaiming_std(fan_in, target_shape, a = math.sqrt(5)):
    """
    Generates a random array with the Kaiming standard deviation.

    Args:
        fan_in: The number of input nodes or channels.
        target_shape: The desired shape of the output array.
        a: A scalar value (optional). By default, it is set to math.sqrt(5).

    Returns:
        An array with random values drawn from a normal distribution with mean 0 and standard deviation calculated
        using the Kaiming initialization technique.

    Raises:
        None.

    Example:
        >>> Kaiming_std(100, (5, 5))
        array([[ 0.01645488, -0.03843136,  0.00841639,  0.04566001, -0.02022473],
               [ 0.01867243,  0.03806507, -0.05415759, -0.00127936,  0.02327622],
               [ 0.08205195,  0.01649465,  0.05024685, -0.05801567, -0.01692097],
               [ 0.01394429,  0.04620856, -0.0164491 ,  0.04990669, -0.01743803],
               [ 0.02244341,  0.05525818, -0.02107648, -0.00010311, -0.03554244]])
    """
    bound = 2.0 / (1 + a * a) / fan_in
    std = math.sqrt(bound)
    return np.random.normal(loc = 0, scale = std, size = target_shape)