import numpy as np

from cnn.utils import NetworkDict, Kaiming_uniform


class Output():
    """

    """
    __name__ = "Output"

    def __init__(self, units, input_shape, activate_fcn = "softmax", theta = None):
        """
        Args:
            units: The number of units in the layer.
            input_shape: The shape of the input data.
            activate_fcn: The activation function to be applied. Default is "softmax".
            theta: The weights for the layer. If None or shape does not match theta_shape, Kaiming_uniform initialization is used.

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
        
        if activate_fcn == "softmax" or activate_fcn == "sigmoid":
            self.flag = True
        else:
            self.flag = False
        
        # 输入，激活项
        self.x = None
        self.a = None
    
    def set_input(self, X):
        self.x = X
        
    def hidden_forward(self, x, theta, activate_fcn):
        """
        Args:
            x: A numpy array of shape (n_samples, n_features) containing the input data.
            theta: A numpy array of shape (n_hidden_units, n_features) representing the weights of the hidden layer.
            activate_fcn: A function that applies the activation function to the input.

        Returns:
            a: A numpy array of shape (n_samples, n_hidden_units) representing the output of the hidden layer.

        Description:
        This method performs the forward pass through the hidden layer of a neural network. It calculates the weighted sum of the input features and the weights of the hidden layer, applies
        * the activation function to the result, and returns the hidden layer output.

        Note that the '@' operator is used for matrix multiplication in this method.
        """
        z = x @ theta.T
        a = activate_fcn(z)

        return a
    
    def forward_propagate(self):
        self.a = self.hidden_forward(self.x, self.theta, self.activate_fcn)
        return self.a
    
    def hidden_backward(self, a, x, error, theta, activate_fcn_gradient, flag):
        """
        Args:
            a: The activation values of the hidden layer (numpy array).
            x: The input values to the hidden layer (numpy array).
            error: The error values from the next layer (numpy array).
            theta: The weight matrix connecting the hidden layer to the next layer (numpy array).
            activate_fcn_gradient: The gradient of the activation function (function).
            flag: A boolean value indicating whether to calculate delta using cross entropy loss + sigmoid/softmax (boolean).

        Returns:
            grad: The gradient of the weights connecting the hidden layer to the next layer (numpy array).
            error_bp: The error to be propagated back to the previous layers (numpy array).

        """
        # calculate delta using cross entropy loss + sigmoid/softmax
        if flag:
            delta = error
        else:
            delta = np.multiply(error, activate_fcn_gradient(a))

        # 计算 grad
        grad = delta.T @ x / x.shape[0]

        # 反向传播
        error_bp = delta @ theta

        return grad, error_bp
    
    def backward_propagate(self, error, lr):
        grad, error_bp = self.hidden_backward(self.a, self.x, error, 
                                              self.theta, self.activate_gradient_fcn, self.flag)
        self.theta -= lr * grad
        return error_bp
    
    def set_flag(self, loss_flag):
        self.flag = self.flag & loss_flag
    
    def summary(self):
        """
        Returns a tuple containing the name of the method, the output shape, and the product of the theta shape.

        :return: A tuple containing the name, output shape, and the product of theta shape.
        :rtype: Tuple[str, Tuple[int], int]
        """
        return self.__name__, self.output_shape, np.prod(self.theta_shape, dtype=np.int32)
    
    def save(self):
        """
        Saves the model's parameters and its string representation.

        Returns:
            Tuple: A tuple containing the initial parameters shape, the model's parameters, and the model's string representation.

        """
        init_params = np.array([self.theta_shape[0]])
        return init_params, self.theta, self._str