import numpy as np

from cnn.utils import NetworkDict, im2col_indices


class MaxPooling2D():
    """
    Class representing a MaxPooling2D layer.

    Usage:
        pool = MaxPooling2D(pool_size, stride=0, input_shape=None, padding='valid')
    """
    __name__ = "MaxPooling2D"

    def __init__(self, pool_size, stride = 0, input_shape = None, padding='valid'):
        """
        Args:
            pool_size (int): The size of the pooling window. It specifies the height and width of the pooling window.
            stride (int, optional): The stride value. It specifies the stride value used for the pooling operation. Default is 0.
            input_shape (tuple, optional): The shape of the input tensor. It specifies the shape of the input tensor in the (C, H, W) format. Default is None.
            padding (str, optional): The padding type. It specifies the type of padding to be applied. Default is 'valid'.

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
        self.max_id = None
    
    def set_input(self, X):
        self.x = X
        
    def max_pool_im2col(self, x, kernel_shape, padding = 0, stride = 0):
        """
        Applies max pooling operation using im2col technique.

        Args:
            x: A 4-D tensor of shape (N, C, H, W), where N is the batch size, C is the number of channels, H is the height and W is the width.
            kernel_shape: A tuple of two integers representing the height and width of the pooling kernel.
            padding: An integer indicating the padding size. Default is 0.
            stride: An integer indicating the stride value. Default is 0.

        Returns:
            A tuple of two 4-D tensors. The first tensor represents the output of the max pooling operation and the second tensor represents the indices of the maximum elements within each pooling
        * region.

        """
        N, C, H, W = x.shape
        kH, kW = kernel_shape
        if stride == 0:
            stride = KH
        out_h = H // stride
        out_w = W // stride

        # im2col
        x_col = im2col_indices(x, kH, kW, 0, stride).reshape(C, kH*kW, -1)

        # 最大池化
        max_id = x_col.argmax(axis = 1)
        z = x_col.max(axis = 1)

        return (z.reshape(C, N, out_h, out_w).transpose(1, 0, 2, 3), 
                max_id.reshape(C, N, out_h, out_w).transpose(1, 0, 2, 3))

    
    def forward_propagate(self):
        self.z, self.max_id = self.max_pool_im2col(self.x, self.pool_shape, 
                                                   self.padding, self.stride)
        return self.z
    
    def max_pool_backward(self, error, max_id, kernel_shape):
        """
        Args:
            error: An array representing the error values.
            max_id: An array representing the indices of the maximum values.
            kernel_shape: A tuple representing the shape of the pooling kernel.

        Returns:
            An array representing the backward pass of the max pooling operation.
        """
        N, out_k, out_h, out_w = error.shape
        KH, KW = kernel_shape

        # Using mask ensures that the expanded max_id is True only when the index position corresponds to the value
        max_id = max_id.repeat(KH, axis = -2).repeat(KW, axis = -1)
        mask = np.tile(np.arange(KH * KW).reshape(KH, KW), [out_h, out_w])

        # delta = error
        return error.repeat(KH, axis = -2).repeat(KW, axis = -1) * (max_id == mask)
    
    def backward_propagate(self, error, lr):
        return self.max_pool_backward(error, self.max_id, self.pool_shape)
    
    def summary(self):
        """
        Returns the name, output shape, and an integer value for the given method.

        :param self: The object instance.
        :return: A tuple containing the name (str), output shape (Tuple[int, int, int]), and an integer value (int).
        """
        return self.__name__, self.output_shape, 0
    
    def save(self):
        """
        Saves the current state of the object.

        Parameters:
            None

        Returns:
            Tuple containing:
                - init_params (numpy.ndarray): Array containing the initial parameters of the object.
                  The array elements are as follows:
                      - 0-index: The first element representing the pool shape of the object.
                      - 1-index: The second element representing the stride of the object.
                - None: Since there are no other values to return, this element will always be None.
                - _str (str): A string representation of the object.

        Example:
            obj = ClassName()
            params, _, string_repr = obj.save()
        """
        init_params = np.array([self.pool_shape[0], self.stride])
        return init_params, None, self._str