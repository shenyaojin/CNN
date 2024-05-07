# Shenyao Jin, shenyaojin@mines.edu
# Dependences: numpy, matplotlib, PIL, os.

from lib.photo import IMAGE
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from cnn.layers import *
from cnn.models import Model, load_model
import time

print("_________________________________________________________________")
print("                   Generating training dataset                   ")
print("=================================================================")
print("                        Reading face files                       ")
def list_files(directory_path):
    # Use the os.walk() function to iterate over each directory in the file system hierarchy
    file_list = []
    for root, dirs, files in os.walk(directory_path):
        # For each directory, iterate over each file
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list
start_path = "./data/faces"
file_list = list_files(start_path)
img_list = []
for file_path in file_list:
    img = IMAGE(file_path)
    img_list.append(img)
print("                               DONE                              ")
print("                        NORMALIZING IMAGES                       ")
# Normalize the figures with z core
for iter in range(len(img_list)):
    img_list[iter].array = img_list[iter].array / 255
    mean_val = np.mean(img_list[iter].array)
    std_val = np.std(img_list[iter].array)
    img_list[iter].array = (img_list[iter].array - mean_val) / std_val

print("                               DONE                              ")
print("                           SELECT IMAGES                         ")
cleaned_imglist = []
count = 0
for iter in range(len(img_list)):
    if img_list[iter].imgsize[0] >= 60:
        cleaned_imglist.append(img_list[iter])
        count += 1
print(f"For cleaned images: the total number of my training data is {count} in {len(img_list)}")
print("                               DONE                              ")
print("                  INTERPOLATE IMAGE TO SAME SIZE...              ")
# interpolate 128 size to 64.
from scipy import ndimage
for iter in range(len(cleaned_imglist)):
    if cleaned_imglist[iter].imgsize[0] == 120:
        cleaned_imglist[iter].array = ndimage.zoom(cleaned_imglist[iter].array, (0.5, 0.5), order=1)
        # update the size
        cleaned_imglist[iter].imgsize = np.shape(cleaned_imglist[iter].array)
    cleaned_imglist[iter].squarize()
print("                   Generating One-hot code                       ")


one_hot_code = []
for iter in range(len(img_list)):
    one_hot_code.append(img_list[iter].gen_onehot_code())
print("                               DONE                              ")
print("                   reading test dataset                       ")
img_list = cleaned_imglist
X_train = []
Y_train = []
X_test = []
Y_test = []
for iter in range(len(img_list)):
    X_train.append(img_list[iter].array)
    Y_train.append(one_hot_code[iter].astype(np.int32))
from lib.test_utils.test_data import load_test_image
X_test, Y_test = load_test_image("./data/training_dataset.pkl")

print("                               DONE                              ")
print("_________________________________________________________________")
print("                   Preparing for training                       ")
print("=================================================================")


X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)
print("_________________________________________________________________")
print("                            INIT CNN                             ")
input_layer = Input((1,64,64))
model = Model(input_layer, "CNN_prototype")
model.add_layer(Conv2D(20, 5, input_shape=(1, 64, 64), activate_fcn="ReLU"))
model.add_layer(AveragePooling2D(2, input_shape=(20, 60, 60)))
model.add_layer(Flatten((20, 30, 30)))
model.add_layer(Dense(400, 18000, activate_fcn = "ReLU"))
model.add_layer(Output(4, 400))
model.compile(0.1, "cross_tropy")
print("                               DONE                              ")

T1 = time.time()
history = model.fit(X_train, Y_train, batch_size = 100, epochs = 80, verbose = 1, shuffle = True)
T2 = time.time()
print('Time Used %s min' % ((T2 - T1) / 60))
from lib.vizutil.visualization import history_show

print("_________________________________________________________________")
print("                            MISFIT VIZ                           ")
print("=================================================================")
history_show(history)
model.save('model_all.h5')
print("                               DONE                              ")

print("_________________________________________________________________")
print("                            MODEL TEST                           ")
print("=================================================================")
print("The accuracy for the test data is ", model.evaluate(X_test, Y_test)[0],
      "The misfit for the test data is ", model.evaluate(X_test, Y_test)[1])

