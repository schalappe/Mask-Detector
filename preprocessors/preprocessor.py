# -*- coding: utf-8 -*-
import cv2
import imutils
from tensorflow.keras.preprocessing.image import img_to_array


class Preprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA, data_format=None):
        # store the target image width, height, and interpolation
        self.width = width
        self.height = height
        self.inter = inter
        # store image
        self.dataFormat = data_format

    def preprocess(self, image):
        # grab the dimensions of the image and then initialize
        # the deltas to use when cropping
        (h, w) = image.shape[:2]
        dW, dH = 0, 0

        # crop
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]

        image = cv2.resize(
            image, (self.width, self.height),
            interpolation=self.inter
        )
        return img_to_array(image, data_format=self.dataFormat) / 255.0
