import numpy as np

from cnn.utils import NetworkDict, Kaiming_uniform


class Dense():
    """
    Dense class

    A class representing a dense (fully connected) layer in a neural network.

    Attributes:
        __name__ (str): The name of the layer.
        input_shape (tuple): The shape of the input data.
        theta_shape (tuple): The shape of the weight matrix.
        theta (ndarray): The weight matrix.
        activate_fcn (method): The activation function.
        activate_gradient_fcn (method): The gradient of the activation function.
        output_shape (tuple): The shape of the output data.
        x (ndarray): The input data.
        a (ndarray): The activation output.

    Methods:
        __init__(units, input_shape, activate_fcn, theta=None):
            Initializes a Dense layer object.
        set_input(X):
            Sets the input data.
        hidden_forward(x, theta, activate_fcn):
            Performs forward propagation for the hidden layer.
        forward_propagate():
            Performs forward propagation.
        hidden_backward(a, x, error, theta, activate_fcn_gradient):
            Performs backpropagation for the hidden layer.
        backward_propagate(error, lr):
            Performs backpropagation.
        summary():
            Returns the layer type, output shape and number of parameters.
        save():
            Returns the initialization parameters, layer parameters and layer configuration.

    """
    __name__ = "Dense"

    def __init__(self, units, input_shape, activate_fcn, theta = None):
        """
        Initializes a Network object.

        Args:
            units (int): The number of units in the layer.
            input_shape (int or tuple): The shape of the input data. If an integer is given, it is assumed to be the number
                                       of features in the input data. If a tuple is given, it should specify the shape of
                                       the input data in the form of ``(num_samples, num_features)``.
            activate_fcn (str): The activation function for the layer.
            theta (numpy.ndarray or None): Optional parameter to initialize the weight matrix of the layer. If None is
                                           given, a weight matrix will be initialized using the Kaiming uniform method.
                                           If a numpy.ndarray is given, it should have the shape ``(units, input_shape)``.

        """
        _dict = NetworkDict(0)
        if type(input_shape) != int:
            input_shape = input_shape[0]
        self.input_shape = (input_shape, )
        self.theta_shape = (units, input_shape)
        if theta is None or theta.shape != self.theta_shape:
            self.theta = Kaiming_uniform(input_shape, units)
        else:
            self.theta = theta
        
        self.activate_fcn = _dict.get_activate_fcn(activate_fcn)
        self.activate_gradient_fcn = _dict.get_activate_gradient_fcn(activate_fcn)
        self.output_shape = (None, units)
        
        self._str = np.array([self.__name__, activate_fcn])

        self.x = None
        self.a = None
    
    def set_input(self, X):
        self.x = X
        
    def hidden_forward(self, x, theta, activate_fcn):
        """
        Args:
            x: The input data of shape (n_samples, n_features).
            theta: The weights of shape (n_hidden_units, n_features).
            activate_fcn: The activation function to be applied on the input data.

        Returns:
            a: The output of the hidden layer after applying the activation function.
        """
        z = x @ theta.T
        a = activate_fcn(z)

        return a
    
    def forward_propagate(self):
        self.a = self.hidden_forward(self.x, self.theta, self.activate_fcn)
        return self.a
    
    def hidden_backward(self, a, x, error, theta, activate_fcn_gradient):
        """
        Args:
            a: The activation output of the hidden layer for the given input batch x.
            x: The input batch for which the hidden layer is being computed.
            error: The error at the output layer, which needs to be backpropagated.
            theta: The weight matrix connecting the hidden layer to the output layer.
            activate_fcn_gradient: The gradient of the activation function used in the hidden layer.

        Returns:
            grad: The gradient of the weight matrix connecting the hidden layer to the input layer.
            error_bp: The error to be backpropagated to the previous layer.

        Description:
        This method is used to perform backward propagation for the hidden layer of a neural network. It calculates the gradient of the weight matrix connecting the hidden layer to the input
        * layer, and also computes the error to be backpropagated to the previous layer.

        The method takes the activation output of the hidden layer, the input batch, the error at the output layer, the weight matrix, and the gradient of the activation function as input parameters
        *. It then calculates the delta by element-wise multiplication of the error and the gradient of the activation function. Next, it computes the grad by performing matrix multiplication
        * of the transposed delta and the input batch, normalized by the number of samples in the batch. Finally, it computes the error_bp by performing matrix multiplication of delta and theta
        *.

        The method returns the grad and error_bp as a tuple.
        """
        delta = np.multiply(error, activate_fcn_gradient(a))

        # calculate gradient
        grad = delta.T @ x / x.shape[0]

        # back propagation
        error_bp = delta @ theta

        return grad, error_bp
    
    def backward_propagate(self, error, lr):
        grad, error_bp = self.hidden_backward(self.a, self.x, error, 
                                              self.theta, self.activate_gradient_fcn)
        self.theta -= lr * grad
        return error_bp
    
    def summary(self):
        """
        Returns the name, output shape, and total number of elements in the theta shape.

        Returns:
            Tuple[str, tuple, int]: A tuple containing the name of the method, the output shape, and the total number of elements in the theta shape.

        """
        return self.__name__, self.output_shape, np.prod(self.theta_shape, dtype=np.int32)
    
    def save(self):
        """
        Save method saves the current state of the object.

        Returns:
            Tuple: A tuple containing the initial parameters, theta values, and the string representation of the object.
        """
        init_params = np.array([self.theta_shape[0]])
        return init_params, self.theta, self._str