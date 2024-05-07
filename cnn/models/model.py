import numpy as np

from cnn.losses import *
from cnn.accuracy import *
from .modelSave import save_model


class Model():
    """
    This class represents a deep learning model.

    Attributes:
        input: The input layer of the model.
        cur_output_shape: The current output shape of the model.
        name: The name of the model.
        layers: The list of layers in the model.
        lr: The learning rate of the model.
        loss_fcn: The loss function of the model.
        loss_fcn_name: The name of the loss function.
        accuracy_fcn: The evaluation function of the model.
        accuracy_fcn_name: The name of the evaluation function.

    Methods:
        __init__(Input_layer, name): Initializes the model with the given input layer and name.
        add_layer(layer): Adds a layer to the model.
        compile(learning_rate, loss_fcn, accuracy_fcn): Sets the learning rate, loss function, and evaluation function of the model.
        forward(): Performs forward propagation in the model.
        backward(error): Performs backward propagation in the model.
        state_zero(): Initializes the state of the recurrent layers in the model to zero.
        fit(x, y, batch_size, epochs, verbose, shuffle): Trains the model on the given data.
        predict(test_data): Predicts the output for the given test data.
        predict_classes(test_data): Predicts the classes for the given test data.
        evaluate(x_test, y_test): Evaluates the model on the given test data.
        summary(): Prints a summary of the model.
        save(filename): Saves the model to a file.
    """
    def __init__(self, Input_layer, name = "cnn"):
        """
        Args:
            Input_layer: The input layer of the network.
            name: The name of the network (default is "cnn").
        """
        self.input = Input_layer
        self.cur_output_shape = Input_layer.input_shape
        self.name = name
        
        self.layers = [] # Layers
        self.lr = 0.0    # learning rate
        
        self.loss_fcn = None     # LOSS function
        self.loss_fcn_name = "_" # LOSS function name
        
        self.accuracy_fcn = None     # Accuracy Function
        self.accuracy_fcn_name = "_" # Accuracy function name
    
    def add_layer(self, layer):
        """
        Args:
            layer: The layer object to be added to the network.

        Raises:
            AssertionError: If the input shape of the layer does not match the current output shape of the network.

        Returns:
            None
        """
        assert layer.input_shape == self.cur_output_shape
        self.layers.append(layer)
        self.cur_output_shape = layer.output_shape[1:]
    
    def compile(self, learning_rate, loss_fcn = "cross_tropy", accuracy_fcn = "categorical_accuracy"):
        """
        Args:
            learning_rate (float): The learning rate for the model.
            loss_fcn (str, optional): The loss function to be used. Defaults to "cross_tropy".
            accuracy_fcn (str, optional): The accuracy function to be used. Defaults to "categorical_accuracy".

        """
        assert learning_rate > 1e-6 and learning_rate < 1
        self.lr = learning_rate
        self.loss_fcn_name = loss_fcn
        self.accuracy_fcn_name = accuracy_fcn
        
        loss_dic = {"cross_tropy": cross_tropy}
        self.loss_fcn = loss_dic[loss_fcn] if loss_fcn in loss_dic else MSE
        
        accuracy_dic = {"categorical_accuracy": categorical_accuracy, "MAE": MAE}
        self.accuracy_fcn = accuracy_dic[accuracy_fcn] if accuracy_fcn in accuracy_dic else MAE
        
        output_layer = self.layers[-1]
        output_layer.set_flag(loss_fcn == "cross_tropy")
            
    def forward(self):
        """
        Forward propagates the input through the layers of the neural network.

        This method initializes the forward propagation process by calling the forward_propagate method of the input layer.
        Then, it iterates through each layer in the network, setting the input of the layer to the output of the previous layer,
        and calling the forward_propagate method of each layer.
        Finally, it returns the output of the last layer.

        Returns:
            The output of the last layer after forward propagation.

        """
        a = self.input.forward_propagate()
        for layer in self.layers:
            layer.set_input(a)
            a = layer.forward_propagate()
        return a
    
    def backward(self, error):
        """
        Args:
            error: The error value obtained from the forward propagation step.

        Returns:
            None
        """
        for layer in self.layers[::-1]:
            error = layer.backward_propagate(error, self.lr)
    
    def state_zero(self):
        for layer in self.layers:
            if layer.__name__ == "Recurrent":
                layer.set_state(np.zeros_like(layer.init_state))
    
    def fit(self, x, y, batch_size = -1, epochs = 1, verbose = 1, shuffle = True):
        """
        Args:
            x: Input data.
            y: Target data.
            batch_size: Number of samples per gradient update. If -1, set to the number of samples in the dataset.
            epochs: Number of times the training algorithm will work through the entire dataset.
            verbose: Verbosity mode. Set 0 for silent, 1 for progress bar, 2 for one line per epoch.
            shuffle: Whether to shuffle the training data at the beginning of each epoch.

        Returns:
            dict: A dictionary containing information about the training process, including accuracy and loss.

        """
        N = x.shape[0]                         # number of sample
        batchs = int(np.ceil(N / batch_size))  # batch number
        index = np.arange(N)                   # INDEX

        if batch_size == -1:
            batch_size = N

        history = {"type": self.accuracy_fcn_name, "accuracy": np.zeros((epochs)), "loss": np.zeros((epochs))}
        print("Model train start.")
        print("=================================================================")
        for i in range(epochs):
            self.state_zero() # Init recurrent
            if shuffle: # shuffle data in every epoch
                np.random.shuffle(index)
                x = x[index]
                y = y[index]
            h = np.zeros(y.shape) # Output
            for j in range(0, N, batch_size):
                k = min(j+batch_size, N)
                Xs = x[j:k]
                ys = y[j:k]
                self.input.set_input(Xs)

                # forward propagation
                a = self.forward()
                h[j:k] = a

                if verbose == 1: # batch log
                    accuracy, _ = self.accuracy_fcn(y[j:k], a)
                    print("batch %8s/%-8s\t%s\tloss: %-10s" % (j//batch_size+1, batchs, accuracy, np.round(self.loss_fcn(a, ys), 6)))

                # backward propagation
                self.backward(a - ys)

            history["loss"][i] = self.loss_fcn(h, y)
            accuracy, history["accuracy"][i] = self.accuracy_fcn(y, h)
            if verbose > 0: # epoch log
                print("_________________________________________________________________")
                print("epoch %8s/%-8s\t%s\tloss: %-10s" % (i+1, epochs, accuracy, np.round(history["loss"][i], 6)))
                print("=================================================================")
        return history
    
    def predict(self, test_data):
        """
        Args:
            test_data: The data used for prediction.

        Returns:
            The prediction result.

        Example:
            test_data = [1, 2, 3]
            prediction = predict(test_data)
        """
        self.input.set_input(test_data)
        return self.forward()
    
    def predict_classes(self, test_data):
        """
        Args:
            self: The instance of the class.
            test_data: The input data for prediction.

        Returns:
            The predicted classes for the input test data.

        """
        return self.predict(test_data).argmax(axis = 1)
    
    def evaluate(self, x_test, y_test):
        """
        Args:
            x_test: The input test data.
            y_test: The corresponding target labels for the test data.

        Returns:
            A tuple containing the accuracy and loss values calculated using the predicted values obtained from the input test data and the target labels.

        Example usage:
            x_test = [...]
            y_test = [...]
            accuracy, loss = evaluate(x_test, y_test)
        """
        a = self.predict(x_test)
        return self.accuracy_fcn(y_test, a)[0], self.loss_fcn(a, y_test)
    
    def summary(self):
        """
            Prints the summary of the model.

            This method prints the model name, followed by a table displaying the layer name, output shape, and number of parameters for each layer in the model. At the end, it also prints the
        * total number of parameters in the model.

            Parameters:
                self (object): The instance of the model.

            Returns:
                None
            """
        total_params = 0
        print("model name: " + self.name)
        print("_________________________________________________________________")
        print("Layer                        Output Shape              Param #   ")
        print("=================================================================")
        for layer in self.layers:
            name, input_shape, params = layer.summary()
            total_params += params
            print("%-29s%-26s%-28s" % (name, input_shape, params))
            print("_________________________________________________________________")
        print("=================================================================")
        print("Total params: %d" % total_params)
        print("_________________________________________________________________")
        
    def save(self, filename):
        """
        Saves the current object to a file.

        Args:
            filename (str): The name of the file to save the object to.

        """
        save_model(filename, self)