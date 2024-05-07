from ..photo import IMAGE
import numpy as np
import pickle

def load_test_image(filepath):
    with open(filepath, "rb") as f:
        img_list = pickle.load(f)
    X_test = []
    Y_test = []
    one_hot_code = []
    for iter in range(len(img_list)):
        one_hot_code.append(img_list[iter].gen_onehot_code())
    for iter in range(len(img_list)):
        X_test.append(img_list[iter].array)
        Y_test.append(one_hot_code[iter].astype(np.int32))

    return X_test, Y_test
