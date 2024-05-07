import h5py
import numpy as np


def save_layer(f, group_name, layer):
    """
    Args:
        f: (h5py.File) The HDF5 file object.
        group_name: (str) The name of the group to create.
        layer: The layer object to save.

    """
    # create group
    cur_group = f.create_group(group_name)
    
    # get parm
    init_params, params, _str = layer.save()
    
    # store string
    dt = h5py.special_dtype(vlen = str)
    ds = cur_group.create_dataset('_str', _str.shape, dtype = dt)
    ds[:] = _str
    
    # store construction parameters
    if init_params is None:
        cur_group.create_dataset('init_params_flag', data = 0)
    else:
        cur_group.create_dataset('init_params_flag', data = 1)
        cur_group.create_dataset('init_params', data = init_params)
        
    # store parameters
    if params is None:
        cur_group.create_dataset('params_flag', data = 0)
    else:
        cur_group.create_dataset('params_flag', data = 1)
        cur_group.create_dataset('params', data = params)


def save_model(filename, model):
    """
    Saves the given model to a file.

    Args:
        filename (str): The name of the file to save the model to.
        model: The model object to be saved.

        The `model` object should have the following attributes:
            - input_shape: The input shape of the model.
            - lr: The learning rate.
            - layer_cnt: The number of layers in the model.
            - name: The name of the model.
            - loss_fcn_name: The name of the loss function used in the model.
            - accuracy_fcn_name: The name of the accuracy function used in the model.

        The `model` object should also have a list of layer objects, where each layer object has the necessary attributes for saving the layer (see `save_layer` function for details).

    Returns:
        None

    Note:
        This method uses the `h5py` library for file handling and data storage.
        The saved model can be loaded using the `load_model` method.

    Example:
        save_model("model.h5", my_model)
    """
    # cr file
    f = h5py.File(filename, "w")

    # save vars
    f.create_dataset("input_shape", data = model.input.input_shape)
    f.create_dataset("lr", data = model.lr)
    f.create_dataset("layer_cnt", data = len(model.layers))

    dt = h5py.special_dtype(vlen = str)
    data = np.array([model.name, model.loss_fcn_name, model.accuracy_fcn_name])
    ds = f.create_dataset('name', data.shape, dtype = dt)
    ds[:] = data

    # save layer
    for i, layer in enumerate(model.layers):
        save_layer(f, f"layer_{i}", layer)
    
    f.close()