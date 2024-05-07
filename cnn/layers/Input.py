class Input():
    """
    Input layer for neural networks.
    """
    __name__ = "Input"

    def __init__(self, input_shape):
        """
        Args:
            input_shape (tuple): The shape of the input data.

        """
        self.input_shape = input_shape

        self.x = None
    
    def set_input(self, X):
        self.x = X
    
    def forward_propagate(self):
        return self.x
    
    def summary(self):
        """
        Returns a tuple containing the name of the method, the input shape, and the sum of the input shape as an integer.

        :return: A tuple containing the method name (str), input shape (list), and sum of input shape (int).
        :rtype: tuple
        """
        return self.__name__, self.input_shape, np.sum(self.input_shape, dtype=np.int32)