# photo IO created by Shenyao, shenyaojin@mines.edu
import os

import numpy as np
from PIL import Image


class IMAGE:
    filepath = "None"

    def __init__(self, file_path=None):
        if file_path is None:
            return None
        self.image = Image.open(file_path)
        self.array = np.array(self.image)
        self.imgsize = np.shape(self.array)
        filename = os.path.basename(file_path)
        name_parts = filename.split(".")[0]
        # Now split the separated filename by underscores
        keywords = name_parts.split("_")
        # tag of file
        self.name = keywords[0]
        self.direction = keywords[1]
        if keywords[2] == "angry":
            self.expression = 0
        elif keywords[2] == "happy":
            self.expression = 1
        elif keywords[2] == "neutral":
            self.expression = 2
        elif keywords[2] == "sad":
            self.expression = 3
        else:
            self.expression = 4

        if keywords[3] == "sunglasses":
            self.sunglass = 1
        else:
            self.sunglass = 0

    def print_expression(self):
        if self.expression == 0:
            print("Angry")
        elif self.expression == 1:
            print("Happy")
        elif self.expression == 2:
            print("Neutral")
        elif self.expression == 3:
            print("Sad")

    def gen_onehot_code(self):
        y = np.zeros(4)
        y[self.expression] = 1
        return y

    def print_key(self):
        for attr in self.__dict__:
            print(f"{attr} = {getattr(self, attr)}")

    def squarize(self):
        height, width = self.imgsize
        if height > width:
            padding = (height - width) // 2
            self.array = np.pad(self.array, ((0, 0), (padding, padding)), mode='constant')
        elif width > height:
            padding = (width - height) // 2
            self.array = np.pad(self.array, ((padding, padding), (0, 0)), mode='constant')
        # update imgsize
        self.imgsize = np.shape(self.array)