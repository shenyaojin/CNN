import numpy as np

from cnn.utils import NetworkDict, im2col_indices


class AveragePooling2D():
    """
    :class: AveragePooling2D

    Represents a class for performing average pooling operation on a 2D input.

    Attributes:
        __name__ (str): Name of the class.
        input_shape (tuple): Shape of the input (C, H, W).
        pool_shape (tuple): Size of the pooling kernel (pool_size, pool_size).
        stride (int): Stride value for the pooling operation.
        padding (str): Type of padding (valid, same, full).
        output_shape (tuple): Shape of the output (None, C, H // stride, W // stride).
        x (None or ndarray): Input data.
        z (None or ndarray): Output data.

    Methods:
        __init__(pool_size, stride=0, input_shape=None, padding='valid'):
            Initializes the AveragePooling2D object.
        set_input(X):
            Sets the input data.
        average_pool_im2col(x, kernel_shape, padding=0, stride=0):
            Implements the efficient implementation of average pooling.
        forward_propagate():
            Performs forward propagation and returns the output.
        average_pool_backward(error, kernel_shape):
            Implements the backward propagation for average pooling.
        backward_propagate(error, lr):
            Performs backward propagation and returns the error.
        summary():
            Returns the layer type, output shape, and number of parameters.
        save():
            Returns the parameters and the layer configuration.

    """
    __name__ = "AveragePooling2D"

    def __init__(self, pool_size, stride = 0, input_shape = None, padding='valid'):
        """
        Args:
            pool_size: An integer representing the size of the pooling window.
            stride: An integer representing the stride value. Default is 0.
            input_shape: A tuple representing the shape of the input data.
            padding: A string representing the padding mode. Default is 'valid'.

        """
        _dict = NetworkDict(pool_size)
        C, H, W = input_shape
        self.input_shape = input_shape
        self.pool_shape = (pool_size, pool_size) 
        self.stride = stride if stride > 0 else pool_size
        self.padding = _dict.get_padding(padding)
        self.output_shape = (None, C, H // self.stride, W // self.stride)
        
        self._str = np.array([self.__name__, padding])

        self.x = None
        self.z = None
    
    def set_input(self, X):
        self.x = X
        
    def average_pool_im2col(self, x, kernel_shape, padding = 0, stride = 0):
        """
        Args:
            x: Numpy array
                The input tensor with shape (N, C, H, W), where
                N - number of samples
                C - number of channels
                H - height of the input tensor
                W - width of the input tensor

            kernel_shape: tuple
                The shape of the pooling kernel/filter. It should contain 2 elements
                (kH, kW), where
                kH - height of the kernel
                kW - width of the kernel

            padding: int, optional
                The number of padding elements to add to each side of the input tensor.
                Default is 0, which means no padding is added.

            stride: int, optional
                The stride of the pooling operation. Default is 0, which means the stride
                is equal to the kernel height (kH).

        Returns:
            numpy array
                The output tensor after applying average pooling operation, with shape
                (N, C, out_h, out_w), where
                out_h - height of the output tensor
                out_w - width of the output tensor

        """
        N, C, H, W = x.shape
        kH, kW = kernel_shape
        if stride == 0:
            stride = KH
        out_h = H // stride
        out_w = W // stride

        # im2col
        x_col = im2col_indices(x, kH, kW, 0, stride).reshape(C, kH*kW, -1)

        z = x_col.mean(axis = 1)

        return z.reshape(C, N, out_h, out_w).transpose(1, 0, 2, 3)

    
    def forward_propagate(self):
        self.z = self.average_pool_im2col(self.x, self.pool_shape, 
                                                   self.padding, self.stride)
        return self.z
    
    def average_pool_backward(self, error, kernel_shape):
        """
        Args:
            error: Numpy array representing the error gradients from the previous layer. Its shape is (batch_size, height, width, channels).
            kernel_shape: Tuple representing the shape of the pooling kernel. Its format is (kernel_height, kernel_width).

        Returns:
            Numpy array representing the output error gradients obtained from the backward pass of the average pooling layer. Its shape is the same as the input 'error' tensor.
        """
        KH, KW = kernel_shape
        # delta = error
        return error.repeat(KH, axis = -2).repeat(KW, axis = -1) / (KH * KW)
    
    def backward_propagate(self, error, lr):
        return self.average_pool_backward(error, self.pool_shape)
    
    def summary(self):
        """Summary method.

        Returns the name, output shape, and an integer.

        Args:
            self: This method does not require any parameters.

        Returns:
            A tuple with the name, output shape, and an integer.
        """
        return self.__name__, self.output_shape, 0
    
    def save(self):
        """
        Save the current state of the object.

        Returns:
            Tuple[np.array, None, str]: A tuple containing the following elements:
                - `init_params` (np.array): The initial parameters for the object, represented as a NumPy array.
                - `None` (None): Placeholder value.
                - `_str` (str): A string representation of the object.
        """
        init_params = np.array([self.pool_shape[0], self.stride])
        return init_params, None, self._str