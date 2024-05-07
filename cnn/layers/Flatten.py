import numpy as np


class Flatten():
    """
    Class for flattening input images.

    Attributes:
        input_shape (tuple): The shape of the input images in the format (C, H, W).
        output_shape (tuple): The shape of the flattened output in the format (None, C*H*W).
        _str (numpy.ndarray): A numpy array used for saving the parameters of the layer.

    Methods:
        __init__(self, input_shape)
            Initializes the Flatten layer with the given input_shape.

        set_input(self, X)
            Sets the input data for the Flatten layer.

        flatten_forward(self, x)
            Flattens the input data.

        forward_propagate(self)
            Performs forward propagation through the Flatten layer.

        flatten_backford(self, error, input_shape)
            Flattens the error in the reverse direction.

        backward_propagate(self, error, lr)
            Performs backward propagation through the Flatten layer.

        summary(self)
            Returns the layer type, output shape, and number of parameters.

        save(self)
            Returns the parameters used to build the convolutional layer and the convolutional kernel parameters.
    """
    __name__ = "Flatten"

    def __init__(self, input_shape):
        """
        Args:
            input_shape: A tuple representing the shape of the input data.

        """
        self.input_shape = input_shape
        self.output_shape = (None, np.prod(input_shape))
        
        self._str = np.array([self.__name__])

        self.x = None
        self.z = None
    
    def set_input(self, X):
        self.x = X
        
    def flatten_forward(self, x):
        """
        Args:
            x: A numpy array of shape (N, C, H, W), where
               N is the batch size, C is the number of channels,
               H is the height of the image, and W is the width of the image.

        Returns:
            A numpy array of shape (N, CHW), where CHW is the flattened
            representation of the input array x, preserving the batch size N.

        Raises:
            None.
        """
        N, C, H, W = x.shape
        self.input_shape = (C, H, W)
        return x.reshape(N, -1)
    
    def forward_propagate(self):
        self.z = self.flatten_forward(self.x)
        return self.z
    
    def flatten_backford(self, error, input_shape):
        """
        Args:
            error: A numpy array representing the error.
            input_shape: A tuple representing the shape of the input.

        Returns:
            A numpy array with the error reshaped.

        Raises:
            None.
        """
        C, H, W = input_shape

        return error.reshape((error.shape[0], C, H, W))
    
    def backward_propagate(self, error, lr):
        return self.flatten_backford(error, self.input_shape)
    
    def summary(self):
        """
        Returns the summary of the method.

        :param self: Reference to the object.
        :return: A tuple containing the name of the method, the output shape, and 0.
        """
        return self.__name__, self.output_shape, 0
    
    def save(self):
        """
        Saves the data and returns tuple of None, None, and self._str.

        :return: Tuple containing None, None, and self._str.
        """
        return None, None, self._str