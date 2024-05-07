import h5py

from cnn.layers import *
from .model import Model
from ..layers.Recurrent import Recurrent

layer_dic = {"Conv2D": Conv2D, "AveragePooling2D": AveragePooling2D, "MaxPooling2D": MaxPooling2D,
            "Flatten": Flatten, "Dense": Dense, "Output": Output, "Recurrent": Recurrent}

def load_layer(f, group_name, input_shape):
    """
    Args:
        f: h5py.File: The h5py file object containing the layer definitions.
        group_name: str: The name of the group within the h5py file that contains the layer definition.
        input_shape: The shape of the input to the layer.

    Returns:
        An instance of the layer class with the specified parameters.

    Raises:
        KeyError: If the specified group name does not exist in the h5py file.
    """
    # specify group
    cur_group = f[group_name]
    
    # paramters
    if cur_group["init_params_flag"][()] == 1:
        init_params = cur_group["init_params"][:].tolist()
    else:
        init_params = []
    init_params.append(input_shape)
        
    _str = cur_group["_str"][:].astype('<U32').tolist()
    layer_class = layer_dic[_str[0]]
    layer_params = init_params + _str[1:]
    if cur_group["params_flag"][()] == 1:
        layer_params.append(cur_group["params"][:])
    
    return layer_class(*layer_params)

def load_model(filename):
    """
    Loads a model from a given file.

    Args:
        filename (str): The path to the file containing the model.

    Returns:
        model: The loaded model.

    Example:
        >>> model = load_model("saved_model.h5")
    """
    f = h5py.File(filename, "r")
    
    # construct model
    input_layer = Input(tuple(f["/input_shape"][:].tolist()))
    name = f["/name"][:].astype('<U32')
    model = Model(input_layer, name[0])
    
    # add layer
    layer_cnt = f["/layer_cnt"][()]
    for i in range(layer_cnt):
        model.add_layer(load_layer(f, f"/layer_{i}", model.cur_output_shape))
    
    # compile
    lr = f["lr"][()]
    model.compile(lr, name[1], name[2])
    
    f.close()
    return model