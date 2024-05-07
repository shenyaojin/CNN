import numpy as np
from scipy.signal import convolve2d


def conv_standard(x, kernel, padding = 0):
    """
    Args:
        x: (ndarray) Input tensor of shape (N, C, H, W).
        kernel: (ndarray) Kernel tensor of shape (out_k, C, kH, kW).
        padding: (int) Number of pixels to pad the input tensor. Default is 0.

    Returns:
        z: (ndarray) Output tensor of shape (N, out_k, out_h, out_w).

    Description:
        This method performs standard convolution on the input tensor using the given kernel. The input tensor is padded with zeros based on the specified padding value. The output tensor
    * is obtained by convolving the input tensor with the kernel using the specified mode. The resulting tensor is returned as the output.

    Example:
        x = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
        kernel = np.array([[[[1, 1], [1, 1]]]])
        padding = 1
        result = conv_standard(x, kernel, padding)
    """
    mode = ["valid", "same", "full"]
    mode = mode[padding]
    # padding
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode = "constant")

    N, C, H, W = x.shape
    out_k, C, kH, kW = kernel.shape
    out_h = (H + 2 * padding - kH) + 1
    out_w = (W + 2 * padding - kW) + 1
    
    # conv
    z = np.zeros((N, out_k, out_h, out_w))
    for i in range(N):
        for j in range(out_k):
            for ci in range(C):
                z[i, j] += convolve2d(x[i, ci], kernel[j, ci, ::-1, ::-1], 
                                      boundary = 'fill', mode = mode, fillvalue = 0)
    return z


def average_pool_standard(x, kernel_shape):
    """
    Args:
        x: The input tensor (N, C, H, W).
        kernel_shape: The shape of the kernel used for pooling (kH, kW).

    Returns:
        The result of average pooling with the specified kernel shape applied to the input tensor.

    Note:
        - The input tensor should have dimensions (N, C, H, W), where:
            - N is the number of samples in the batch
            - C is the number of channels
            - H is the height of the input
            - W is the width of the input.
        - The kernel_shape should have dimensions (kH, kW), where:
            - kH is the height of the kernel
            - kW is the width of the kernel.
        - The output shape is determined by dividing the input height and width by the corresponding kernel dimensions.
        - The average pooling operation is performed by iterating through the input tensor with stride equal to kernel height and width.
        - The output is obtained by averaging the values within each pooling window.

    Example:
        x = np.random.rand(2, 3, 6, 6)
        kernel_shape = (2, 2)
        output = average_pool_standard(x, kernel_shape)
    """

    N, C, H, W = x.shape
    kH, kW = kernel_shape
    out_h = H // kH
    out_w = W // kW
    
    # avg pooling
    z = np.zeros((N, C, out_h, out_w))
    for i in range(kH):
        for j in range(kW):
            z += x[:, :, i::kH, j::kW]
    
    return z/(kH * kW)


def max_pool_standard(x, kernel_shape):
    """
    Performs standard max pooling on the input tensor.

    Args:
        x: Input tensor of shape (N, C, H, W), where
           N - batch size,
           C - number of channels,
           H - height of input feature map,
           W - width of input feature map.
        kernel_shape: Shape of the pooling kernel (kH, kW), where
                      kH - height of the kernel,
                      kW - width of the kernel.

    Returns:
        A tuple containing two tensors:
        - z: Result of max pooling, tensor of shape (N, C, out_h, out_w), where
             out_h - height of the output feature map,
             out_w - width of the output feature map.
        - max_id: The indices of the maximum values in input tensor, tensor of
                  shape (N, C, out_h, out_w), where each element represents the
                  index of maximum value found for the corresponding input feature.

    Examples:
        >>> x = np.random.randn(2, 3, 6, 6)
        >>> kernel_shape = (2, 2)
        >>> z, max_id = max_pool_standard(x, kernel_shape)
    """
    N, C, H, W = x.shape
    kH, kW = kernel_shape
    out_h = H // kH
    out_w = W // kW
    
    # max pooling
    z = np.zeros((N, C, out_h, out_w))
    max_id = np.zeros((N, C, out_h, out_w), dtype = np.int32)
    for i in range(kH):
        for j in range(kW):
            target = x[:, :, i::kH, j::kW]
            mask = target > z
            max_id = max_id * (~mask) + mask * (i * kH + j)
            z = z * (~mask) + mask * target
    
    return z, max_id


def conv_bp_standard(x, z, error, kernel, activate_fcn_gradient):
    """
    Args:
        x: A numpy array representing the input data. The shape should be (N, C, H, W), where N is the batch size, C is the number of channels, H is the input height, and W is the input
    * width.

        z: A numpy array representing the output of the convolution layer. The shape should be (N, out_k, H_out, W_out), where out_k is the number of output channels, H_out is the output
    * height, and W_out is the output width.

        error: A numpy array representing the backpropagated error. The shape should be (N, out_k, H_out, W_out).

        kernel: A numpy array representing the convolution kernel. The shape should be (out_k, C, KH, KW), where KH is the kernel height and KW is the kernel width.

        activate_fcn_gradient: A function that computes the gradient of the activation function. It takes a numpy array as input and returns a numpy array of the same shape.

    Returns:
        grad: A numpy array representing the gradient of the convolution kernel. The shape is the same as the kernel shape (out_k, C, KH, KW).

        error_bp: A numpy array representing the backpropagated error to the previous layer. The shape is the same as the input shape (N, C, H, W).
    """
    N, C, H, W = x.shape
    out_k, C, KH, KW = kernel.shape
    
    # calculate delta
    delta = np.multiply(error, activate_fcn_gradient(z))
    
    # calculate grad
    grad = np.zeros((out_k, C, KH, KW))
    for i in range(N):
        for j in range(out_k):
            for ci in range(C):
                grad[j, ci] += convolve2d(x[i, ci], delta[i, j][::-1, ::-1], mode = "valid")
    grad /= N
    
    # back prop
    error_bp = np.zeros((N, C, H, W))
    for i in range(N):
        for j in range(out_k):
            for ci in range(C):
                error_bp[i, ci] += convolve2d(delta[i, j], kernel[j, ci][::-1, ::-1], 
                                         boundary = 'fill', mode = "full", fillvalue = 0)
    
    return grad, error_bp


def average_pool_backward_standard(error, kernel_shape):
    """
    Args:
        error: A tensor representing the error propagated back from the forward pass of the average pooling layer.
        kernel_shape: A tuple of two integers representing the shape of the pooling kernel.

    Returns:
        A tensor representing the backpropagated error for the average pooling layer.

    Raises:
        None
    """
    KH, KW = kernel_shape
    # delta = error
    return error.repeat(KH, axis = -2).repeat(KW, axis = -1) / (KH * KW)


def max_pool_backward_standard(error, max_id, kernel_shape):
    """
    Args:
        error (ndarray): The error array from the previous layer. It has shape (N, out_k, out_h, out_w).
        max_id (ndarray): The id array of max pooling from the forward pass. It has shape (N, out_k).
        kernel_shape (tuple): The shape of the pooling kernel. It has the format (KH, KW).

    Returns:
        error_bp (ndarray): The error backward propagated to the previous layer. It has shape (N, out_k, KH * out_h, KW * out_w).

    This method performs the backward pass of a standard max pooling layer. It computes the error backward propagated to the previous layer.

    The error backward propagation is done by assigning the errors in the error array to the corresponding locations in the error_bp array. The positions for assignment are determined using
    * the max_id array, which contains the indices of the max values in the forward pass.

    The error_bp array is initialized with zeros and has the same shape as the input feature map before pooling. Each value in the error array is assigned to the corresponding position in
    * the error_bp array based on the indices in the max_id array.

    Example usage:
        error = np.random.randn(N, out_k, out_h, out_w)
        max_id = np.random.randint(0, KH*out_h*KW*out_w, size=(N, out_k))
        kernel_shape = (KH, KW)
        error_bp = max_pool_backward_standard(error, max_id, kernel_shape)
    """
    N, out_k, out_h, out_w = error.shape
    KH, KW = kernel_shape
    
    error_bp = np.zeros((N, out_k, KH * out_h, KW * out_w))
    for i in range(N):
        for j in range(out_k):
            row = max_id[i, j] // KH + np.arange(out_h).reshape(-1, 1) * KH
            col = max_id[i, j] % KH + np.arange(out_w).reshape(1, -1) * KW
            error_bp[i, j, row, col] = error[i, j]

    return error_bp