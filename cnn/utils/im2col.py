import numpy as np

def get_im2col_indices(x_shape, field_height, field_width, padding=0, stride=1):
    """
    Args:
        x_shape: A tuple representing the shape of the input tensor (N, C, H, W).
        field_height: An integer representing the height of the convolutional filter.
        field_width: An integer representing the width of the convolutional filter.
        padding: An integer representing the number of rows/columns to be padded around the input tensor.
            Default is 0.
        stride: An integer representing the stride of the convolution operation. Default is 1.

    Returns:
        A tuple containing three numpy arrays: (k, i, j).
            - k: A numpy array of shape (M, 1), where M is the total number of elements in the convolutional filter.
                Represents the channel indices of each element in the filter.
            - i: A numpy array of shape (M, 1), where M is the total number of elements in the convolutional filter.
                Represents the row indices of each element in the filter.
            - j: A numpy array of shape (M, 1), where M is the total number of elements in the convolutional filter.
                Represents the column indices of each element in the filter.

    Note:
        This method is used to generate row, column, and channel indices for the im2col operation in convolutional neural networks.
    """
    N, C, H, W = x_shape

    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=0, stride=1):
    """
    Args:
        x: Array of shape (batch_size, C, H, W) representing the input image.
        field_height: Integer representing the height of the receptive field/kernel.
        field_width: Integer representing the width of the receptive field/kernel.
        padding: Integer representing the amount of padding applied to the input image. Default is 0.
        stride: Integer representing the stride used for sliding the kernel over the input image. Default is 1.

    Returns:
        Array of shape (field_height * field_width * C, oH * oW) containing the flattened receptive fields.
        The value oH represents the output height and oW represents the output width. The output height and width
        are calculated based on the input image size, receptive field size, padding, and stride values.
    """
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    return cols.transpose(1, 0, 2).reshape(field_height * field_width * C, -1)

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=0, stride=1):
    """
    Args:
        cols: A 2D numpy array representing the reshaped input data (output of im2col_indices function)
        x_shape: A tuple (N, C, H, W) representing the shape of the input data before reshaping
        field_height: An integer representing the height of the receptive field (default is 3)
        field_width: An integer representing the width of the receptive field (default is 3)
        padding: An integer representing the amount of padding applied to the input data (default is 0)
        stride: An integer representing the stride used for sliding the receptive field (default is 1)

    Returns:
        A 4D numpy array representing the input data reshaped back to its original shape after applying col2im operation

    """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]