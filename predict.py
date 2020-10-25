# -*- coding: utf-8 -*-
"""
    Sript use to predict if an image contain a mask
"""
import cv2
import argparse
import tensorflow as tf
from preprocessors import Preprocessor

# argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path of image to process')
args = vars(ap.parse_args())
# model
model = tf.keras.models.load_model('./model/best_nasnet.h5')
# preprocessor
pp = Preprocessor(224, 224)

# load input image
image = cv2.imread(args['image'])
img = pp.preprocess(image)
img = img.reshape(1, 224, 224, 3)
prob = model.predict(img)[0][0]

if prob > 0.5:
    print("There are a mask in this image: {}".format(args['image']))
else:
    print("There aren't a mask in this image: {}".format(args['image']))
