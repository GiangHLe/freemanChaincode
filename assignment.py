import numpy as np
import math
import cv2
from matplotlib import pyplot as mlt
from __future__ import print_function, division

class FreemanChainCode(object):
    def __init__(self, image_path, **kwargs):
        """
        Required arguments:
        - image_path: the location of image, remember '/' with Linux and '\\' with Windows
        Optional arguments:
        - direction: number of direction of chaincode
        Default is 8
        - object_location: list of tuple elements location of object in image
        Default is [(20,218),(20,218)]
        - threshold: choose a threshold for detect object clearly with environment
        """
        self.path = image_path
        self.direction = kwargs.pop('direction', 8)
        self.object_location = kwargs.pop('location', [(20,218),(20,218)])
        self.thresh = kwargs.pop('threshold', 60)
        self.kernel_size = kwargs.pop('kernel_size', 0)

    def _preprocessing(self):
        image = cv2.imread(self.path,0)
        boudary = self.object_location
        image = image[boudary[0][0]:boudary[0][1],boudary[1][0]:boudary[1][1]]
        if image[0,0] == 255:
            ret, img = cv2.threshold(image, self.thresh,255, cv2.THRESH_BINARY_INV)
        else:
            ret.img = cv2.threshold(image, self.thresh, 255, cv2.THRESH_BINARY)
        return img
    
    def _start_point(self):
        for i, row in enumerate(img):
            for j, value in enumerate(row):
                if value == 255:
                    start_point = (i, j)
                    break
            else:
                continue
            break
        return start_point
    def _closing(self):
        kernel = np.ones((10,10),np.uint8)
        img = _preprocessing()
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return closing
    def _opening(self):
        kernel = np.ones((10,10),np.uint8)
        img = _preprocessing()
        opening = cv.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    
    