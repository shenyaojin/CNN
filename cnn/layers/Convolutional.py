import numpy as np

from cnn.utils import NetworkDict, im2col_indices, Kaiming_std


class Conv2D():

    __name__ = "Conv2D"

    def __init__(self, filters, kernel_size, stride = 1,
                 input_shape = None, padding = "valid", activate_fcn = "ReLU",
                 kernel = None):
        """
        Args:
            filters: Number of filters in the convolution. Determines the depth of the output feature map.
            kernel_size: Size of the kernel/filter used in the convolution. Should be an odd number.
            stride (optional): Stride value for the convolutional operation. Default is 1.
            input_shape (optional): Shape of the input tensor. Should be a tuple of 3 values: (channels, height, width).
            padding (optional): Padding mode for the convolutional operation. Default is "valid".
            activate_fcn (optional): Activation function to be applied to the output feature map. Default is "ReLU".
            kernel (optional): Predefined kernel/filter to be used in the convolution. Should have the shape
                (filters, input_shape[0], kernel_size, kernel_size).

        """
        _dict = NetworkDict(kernel_size)
        C, H, W = input_shape
        self.kernel_shape = (filters, input_shape[0], kernel_size, kernel_size)
        if kernel is None or kernel.shape != self.kernel_shape:
            self.kernel = Kaiming_std(np.prod(input_shape), self.kernel_shape)
        else:
            self.kernel = kernel
        
        self.activate_fcn = _dict.get_activate_fcn(activate_fcn)
        self.activate_gradient_fcn = _dict.get_activate_gradient_fcn(activate_fcn)
        
        self.input_shape = input_shape
        self.stride = stride
        self.padding = _dict.get_padding(padding)
        self.output_shape = (None, filters, (H + 2 * self.padding - kernel_size) // stride + 1, (W + 2 * self.padding - kernel_size) // stride + 1)
        
        self._str = np.array([self.__name__, padding, activate_fcn])

        self.x = None
        self.a = None
    
    def set_input(self, X):
        self.x = X
        
    def conv_im2col(self, x, kernel, padding = 0, stride = 1):
        """
        Args:
            x: Input tensor with shape (N, C, H, W), where N is the batch size, C is the number of channels, H is the height, and W is the width.
            kernel: Convolutional kernel with shape (out_k, C, kH, kW), where out_k is the number of output channels, kH is the kernel height, and kW is the kernel width.
            padding: Amount of padding to be added to the input tensor (default = 0).
            stride: Stride value for the convolution operation (default = 1).

        Returns:
            Convolution result tensor with shape (out_k, N, out_h, out_w), where out_h and out_w are the height and width of the output tensor, respectively.

        """
        N, C, H, W = x.shape
        out_k, C, kH, kW = kernel.shape
        out_h = (H + 2 * padding - kH) // stride + 1
        out_w = (W + 2 * padding - kW) // stride + 1

        # im2col
        x_col = im2col_indices(x, kH, kW, padding, stride)
        kernel_col = kernel.reshape(out_k, -1)

        # conv
        z = kernel_col @ x_col

        return z.reshape(out_k, N, out_h, out_w).transpose(1, 0, 2, 3)
    
    def forward_propagate(self):
        z = self.conv_im2col(self.x, self.kernel, self.padding, self.stride)
        self.a = self.activate_fcn(z)
        return self.a
    
    def conv_bp(self, x, a, error, kernel, activate_fcn_gradient):
        """
        Args:
            x: The input tensor of shape (N, C, H, W).
            a: The intermediate activation tensor of shape (N, C, H, W).
            error: The error tensor of shape (N, C, H, W).
            kernel: The kernel tensor of shape (C, C, K, K).
            activate_fcn_gradient: The gradient of the activation function.

        Returns:
            grad: The gradient tensor of the convolution operation with respect to the input tensor.
            error_bp: The error tensor propagated backwards through the convolution operation.

        """
        N, C, H, W = x.shape

        # calculate Delta
        delta = np.multiply(error, activate_fcn_gradient(a))

        # calculate gradient
        grad = self.conv_im2col(x.transpose(1, 0, 2, 3), 
                           delta.transpose(1, 0, 2, 3)).transpose(1, 0, 2, 3) / N

        # back propagation
        error_bp = self.conv_im2col(delta, kernel.transpose(1, 0, 2, 3), 
                                    padding = kernel.shape[-1]-1)

        return grad, error_bp
    
    def backward_propagate(self, error, lr):
        grad, error_bp = self.conv_bp(self.x, self.a, error, 
                                      self.kernel, self.activate_gradient_fcn)
        self.kernel -= lr * grad
        return error_bp
    
    def summary(self):
        """
        Returns the name of the method, the output shape, and the product of the kernel shape.

        :return: A tuple containing the name, output shape, and product of kernel shape.
        :rtype: tuple
        """
        return self.__name__, self.output_shape, np.prod(self.kernel_shape, dtype=np.int32)
    
    def save(self):
        """
        Saves the current state of the object.

        Returns:
            Tuple: A tuple containing the following parameters:
                - init_params (np.array): An array containing the initial parameters of the object. The array has three elements
                    representing the kernel shape, the last dimension of the kernel, and the stride.
                - kernel (np.array): An array representing the current kernel of the object.
                - _str (str): A string representing the current state of the object.
        """
        init_params = np.array([self.kernel_shape[0], self.kernel_shape[-1], self.stride])
        return init_params, self.kernel, self._str