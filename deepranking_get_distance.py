# coding: utf-8

import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="Path to the deep ranking model")

ap.add_argument("-i1", "--image1", required=True,
    help="Path to the first image")

ap.add_argument("-i2", "--image2", required=True,
    help="Path to the second image")

args = vars(ap.parse_args())

if not os.path.exists(args['model']):
    print "The model path doesn't exist!"
    exit()

if not os.path.exists(args['image1']):
    print "The image 1 path doesn't exist!"
    exit()

if not os.path.exists(args['image2']):
    print "The image 2 path doesn't exist!"
    exit()

args = vars(ap.parse_args())

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from skimage import transform
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Embedding

def convnet_model_():
    vgg_model = VGG16(weights=None, include_top=False)
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Lambda(lambda  x_: K.l2_normalize(x,axis=1))(x)
    convnet_model = Model(inputs=vgg_model.input, outputs=x)
    return convnet_model

def deep_rank_model():
 
    convnet_model = convnet_model_()
    first_input = Input(shape=(224,224,3))
    first_conv = Conv2D(96, kernel_size=(8, 8),strides=(16,16), padding='same')(first_input)
    first_max = MaxPool2D(pool_size=(3,3),strides = (4,4),padding='same')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(first_max)

    second_input = Input(shape=(224,224,3))
    second_conv = Conv2D(96, kernel_size=(8, 8),strides=(32,32), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7,7),strides = (2,2),padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(second_max)

    merge_one = concatenate([first_max, second_max])

    merge_two = concatenate([merge_one, convnet_model.output])
    emb = Dense(4096)(merge_two)
    l2_norm_final = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)

    final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

    return final_model


model = deep_rank_model()

# for layer in model.layers:
#     print (layer.name, layer.output_shape)

model.load_weights(args['model'])

image1 = load_img(args['image1'])
image1 = img_to_array(image1).astype("float64")
image1 = transform.resize(image1, (224, 224))
image1 *= 1. / 255
image1 = np.expand_dims(image1, axis = 0)

embedding1 = model.predict([image1, image1, image1])[0]

image2 = load_img(args['image2'])
image2 = img_to_array(image2).astype("float64")
image2 = transform.resize(image2, (224, 224))
image2 *= 1. / 255
image2 = np.expand_dims(image2, axis = 0)

embedding2 = model.predict([image2,image2,image2])[0]

distance = sum([(embedding1[idx] - embedding2[idx])**2 for idx in range(len(embedding1))])**(0.5)

print (distance)

