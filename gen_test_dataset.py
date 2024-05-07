import numpy as np
from lib.photo import IMAGE
import os
from PIL import Image
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


for iter in range(len(img_list)):
    img_list[iter].array = img_list[iter].array / 255
    mean_val = np.mean(img_list[iter].array)
    std_val = np.std(img_list[iter].array)
    img_list[iter].array = (img_list[iter].array - mean_val) / std_val

# select images
cleaned_imglist = []
count = 0
for iter in range(len(img_list)):
    if img_list[iter].imgsize[0] < 60:
        cleaned_imglist.append(img_list[iter])
        count += 1
print(f"For Low resolution images: the total number of my training data is {count} in {len(img_list)}")
from scipy import ndimage
for iter in range(len(cleaned_imglist)):
    if cleaned_imglist[iter].imgsize[0] == 30:
        cleaned_imglist[iter].array = ndimage.zoom(cleaned_imglist[iter].array, (2, 2), order=1)
        # update the size
        cleaned_imglist[iter].imgsize = np.shape(cleaned_imglist[iter].array)
    cleaned_imglist[iter].squarize()

print("Saving train datasets...")
import pickle
with open("./data/training_dataset.pkl", "wb") as handle:
    pickle.dump(cleaned_imglist, handle, protocol=pickle.HIGHEST_PROTOCOL)