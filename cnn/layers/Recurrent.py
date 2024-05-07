import numpy as np

from cnn.utils import NetworkDict, Kaiming_uniform

# DO NOT use the code in this file, under testing -- JIN.
class Recurrent():
    """
    class Recurrent():
        __name__ = "Recurrent"

        def __init__(self, units, input_shape, activate_fcn, theta = None):
            '''
            Initializes Recurrent layer object.

            Args:
                units (int): Output dimension size.
                input_shape (int): Dimensionality of the input data to this layer.
                activate_fcn (string): Activation function.
                theta (?): Specified parameters.
            '''
            ...

        def set_state(self, state):
            '''
            Sets the initial state of the layer.

            Args:
                state: Initial state of the layer.
            '''
            ...

        def set_input(self, X):
            '''
            Sets the input data of the layer.

            Args:
                X: Input data.
            '''
            ...

        def forward_propagate(self):
            '''
            Performs forward propagation in the layer.

            Returns:
                array
    """
    __name__ = "Recurrent"

    def __init__(self, units, input_shape, activate_fcn, theta = None):
        """
        Args:
            units: The number of units in the layer.
            input_shape: The shape of the input for the layer. Can be an integer or a tuple of integers.
            activate_fcn: The activation function to be used in the layer.
            theta: (optional) The initial weights and biases of the layer. If not specified or if the shape is incorrect, the layer will be initialized with Kaiming uniform distribution.

        """
        _dict = NetworkDict(0)
        if type(input_shape) != int:
            input_shape = input_shape[0]
        self.input_shape = (input_shape, )
        self.U_shape = (units, input_shape)
        self.W_shape = (units, units)
        # INIT
        if theta is None or theta.shape != (units, input_shape + units + 1):
            self.U = Kaiming_uniform(input_shape, units)
            self.W = Kaiming_uniform(units, units)
            # initial status
            self.init_state = np.zeros((1, units))
        else:
            self.U = theta[:, :input_shape]
            self.W = theta[:, input_shape:-1]
            self.init_state = theta[:, -1]
        
        self.activate_fcn = _dict.get_activate_fcn(activate_fcn)
        self.activate_gradient_fcn = _dict.get_activate_gradient_fcn(activate_fcn)
        self.output_shape = (None, units)
        
        self._str = np.array([self.__name__, activate_fcn])
        
        # input and activate
        self.x = None
        self.a = None
    
    def set_state(self, state):
        self.init_state = state
        self.a = None
        
    def set_input(self, X):
        self.x = X
        
    def recurrent_forward(self, x, U, W, S_pre, activate_fcn):
        """
        Args:
            x: The input value(s) for the recurrent forward pass. It can be a single value or a matrix.
            U: The weight matrix connecting the input to the hidden state.
            W: The weight matrix connecting the previous hidden state to the current hidden state.
            S_pre: The previous value(s) of the hidden state.
            activate_fcn: The activation function to be applied to the input of the hidden state.

        Returns:
            The output value(s) after applying the activation function.

        """
        return activate_fcn(x @ U.T + S_pre @ W.T)
    
    def forward_propagate(self):
        batch_size = self.x.shape[0]
        units = self.output_shape[1]
        output = np.zeros((batch_size, units))

        if self.a is not None:
            self.init_state = self.a[-1]
        
        cur_state = self.init_state # initial condition
        for i in range(batch_size):
            cur_state = self.recurrent_forward(self.x[i], self.U, self.W, 
                                               cur_state, self.activate_fcn)
            output[i] = cur_state
        
        self.a = output
        return self.a
    
    def recurrent_backward(self, h, a, S_lst, error, U, W, activate_fcn_gradient):
        """
        Recursively propagates the error gradient backwards from the output layer to the input layer.

        Args:
            h: The hidden layer activations at each time step, shape (m, t).
            a: The input layer activations at each time step, shape (m, t).
            S_lst: The hidden layer states at each time step, shape (m, t).
            error: The error gradient at each time step, shape (m, t).
            U: The weight matrix connecting the hidden layer to the output layer, shape (n_h, n_y).
            W: The weight matrix connecting the hidden layer to itself, shape (n_h, n_h).
            activate_fcn_gradient: The gradient of the activation function used in the hidden layer.

        Returns:
            grad_U: The gradient of the weight matrix U, shape (n_h, n_y).
            grad_W: The gradient of the weight matrix W, shape (n_h, n_h).
            error_bp: The error gradient propagated to the previous time steps, shape (m, t).

        """
        m, t = error.shape

        delta = np.zeros((m, t))
        v_gradient = activate_fcn_gradient(h)
        error_bp_recurrent = np.zeros((1, t))
        for i in range(m - 1, -1, -1):
            t_error = error[i] + error_bp_recurrent
            delta[i] = np.multiply(t_error, v_gradient[i])
            error_bp_recurrent = delta[i] @ W

        # Calculate gradient
        grad_U = delta.T @ a 
        grad_W = delta.T @ S_lst

        # back propagation
        error_bp = delta @ U

        return grad_U, grad_W, error_bp
    
    def backward_propagate(self, error, lr):
        self.init_state = self.init_state.reshape(1, -1)
        S_lst = np.concatenate((self.init_state, self.a[:-1]), axis = 0)
        grad_U, grad_W, error_bp = self.recurrent_backward(self.a, self.x, S_lst, error,
                                                        self.U, self.W, self.activate_gradient_fcn)
        self.U -= lr * grad_U
        self.W -= lr * grad_W
        return error_bp
    
    def summary(self):
        """
        Returns the name, output shape, and total number of parameters for the given method.

        :return: A tuple containing the name, output shape, and total number of parameters.
        :rtype: tuple
        """
        return self.__name__, self.output_shape, np.prod(self.U_shape, dtype=np.int32) + np.prod(self.W_shape, dtype=np.int32)
    
    def save(self):
        """
        Save the current state of the object.

        Returns:
            Tuple: init_params (numpy.array), concatenated arrays of U, W, and init_state (numpy.array), and _str (str).

        """
        init_params = np.array([self.U_shape[0]])
        if self.a is not None:
            self.init_state = self.a[-1]
        return init_params, np.concatenate((self.U, self.W, self.init_state.reshape(-1, 1)), axis = 1), self._str