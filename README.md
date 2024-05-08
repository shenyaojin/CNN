# Convolutional Neural Network (numpy)
The Python implementation of the Convolutional Neural Network(CNN) uses numpy. Acknowledgement: 贝亚子零号.

# Code structure

./cnn

├── \__init__.py

├── accuracy

│  ├── \__init__.py

│  └── accuracy.py

├── activations

│  ├── \__init__.py

│  └── activate_fcn.py

├── layers

│  ├── AveragePooling.py

│  ├── Convolutional.py

│  ├── Dense.py

│  ├── Flatten.py

│  ├── Input.py

│  ├── MaxPooling.py

│  ├── Output.py

│  ├── Recurrent.py

│  └── \__init\__.py

├── lossesß

│  ├── \__init__.py

│  └── loss.py

├── models

│  ├── \__init__.py

│  ├── model.py

│  ├── modelLoad.py

│  └── modelSave.py

└── utils

​    ├── \__init__.py

​    ├── im2col.py

​    ├── nndict.py

​    ├── standard.py

​    └── weight_initialization.py

accuracy:

accuracy.py: This module computes the accuracy metrics for the model's outputs compared to the true labels, which is critical for assessing your neural network's performance.
activations:

activate_fcn.py: This file hosts key activation functions such as ReLU, Sigmoid, and Tanh. These functions are crucial as they introduce non-linearity into the model, enabling the learning of complex patterns.

layers:

AveragePooling.py: Establishes the average pooling layer, essential for downsampling input representations and simplifying the input data without losing critical information.

Convolutional.py: Implements convolutional layers, the fundamental building blocks of CNNs that perform convolutions to capture spatial hierarchies.

Dense.py: Provides the implementation for dense layers, where every input node is connected to every output node, pivotal for learning high-level patterns.

Flatten.py: Flattens multi-dimensional inputs into a one-dimensional array, bridging convolutional layers and dense layers.
Input.py: Manages input data specification and preparation, setting the stage for data processing.

MaxPooling.py: Implements max pooling layers that reduce the spatial dimensions of the input, enhancing the detection of features in small regions of the input.

Output.py: Defines the output layer that uses specific activation functions tailored to the task, such as softmax for classification.

Recurrent.py: Incorporates recurrent layers, but I have not tested it yet, and I didn't use it in my code.

loss.py: This file contains various loss functions like cross-entropy and mean squared error, which are essential for quantifying the difference between predicted outputs and actual values.

models:

model.py: Constructs the neural network architecture, defining how layers are stacked and data flows through the model.
modelLoad.py: Contains functions for loading models from .h5 files, ensuring that pre-trained systems can be utilized effectively.
modelSave.py: Provides mechanisms for saving the trained models or weights, facilitating the preservation and reuse of models.

utils:

im2col.py: Converts image blocks into columns, a critical preprocessing step for efficient convolution operations.
nndict.py: Manages network parameters or metadata, supporting organized and accessible data handling.
standard.py: Includes standard functions or constants used throughout the project, ensuring consistency and reliability.
weight_initialization.py: Offers various strategies for weight initialization like Xavier and He, which are fundamental for the effective training of neural networks.

lib: 

photo: Includes the I/O and some basic methods to handle the pgm format data.
test_utils: Includes the methods to generate test datasets.
vizutil: Provides visualization methods to show the CNN results, including plot the loss curve and accuracy for both training data and testing datasets.

# How to run the code?

(1) Open Pycharm, and navigate to "gen_test_data.py", and run it. Then run the "train_with_all.py", "train_with_glass.py" and "train_without_glass.py", respectively. Then you could get all the figures in my report. The first figure is in "data_vis.ipynb" which is located at the root path.

(2) For the command line: a. navigate to the "prototype" folder. b. run the following code "python gen_test_data.py", "python train_with_all.py", "python train_with_glass.py" and "python train_without_glass.py", respectively.
