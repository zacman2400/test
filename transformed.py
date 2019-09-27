# coding: utf-8

# In[1]:
import numpy as np
import hist_match as hist_match
import tensorflow as tf
#import tensorflow.keras as keras

import math


import matplotlib
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
'''
from keras.models import Input, Model, Sequential
from keras import optimizers
from keras import layers
from keras.layers import SeparableConv2D, Activation, PReLU, Add, Subtract, LocallyConnected2D, Conv2D, MaxPooling2D, \
    Flatten, Dense, Conv3D, BatchNormalization, UpSampling2D, Concatenate, concatenate, BatchNormalization, add
from keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
from keras.layers.advanced_activations import LeakyReLU
'''
import scipy
from scipy import ndimage

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


from skimage import exposure

import scipy.misc
'''
from keras.optimizers import Adamax
from keras_gradient_noise import add_gradient_noise
'''
from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
import tensorflow as tf
from scipy import ndimage
import tensorflow as tf
from spatial_transformer import ElasticTransformer, AffineTransformer
import numpy as np
import scipy.misc
from spatial_transformer_Copy import ElasticTransformer as ET

def generator(x,y,z,isTrain=True,reuse=False): #registration module 1
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        comb=tf.concat([x,ref1],axis=3)
        x = tf.layers.conv2d(comb, filters=16, kernel_size=3, strides=1, padding='same')
        x = tf.nn.elu(x)
        # x=tf.contrib.layers.batch_norm(x)
        x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, padding='same')
        x = tf.nn.elu(x)
        # x=tf.contrib.layers.batch_norm(x)
        x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, padding='same')
        x = tf.nn.elu(x)
        # x=tf.contrib.layers.batch_norm(x)
        x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, padding='same')
        x = tf.nn.elu(x)
        # x=tf.contrib.layers.batch_norm(x)
        x = tf.layers.flatten(x)
        # x = tf.layers.dense(x,1024,activation='relu')
        deform = tf.layers.dense(x,1250, activation='linear')
        outsize = (192), (144)
        with tf.Session() as sess:
            with tf.device("/gpu:0"):
                with tf.variable_scope('spatial_transformer'):
                   # mov1_matched=hist_match.hist_match(mov1,ref1)
                    stl = ElasticTransformer(outsize)
                    result=stl.transform(mov1,deform)
        x=tf.keras.layers.GaussianNoise(.2)(result)
        l=tf.concat([x,sobel1],axis=3)




       # x=tf.contrib.layers.instance_norm(x) #image translation module 1: mov1'-->ref1_guess
        x = tf.layers.conv2d(l, filters=64, kernel_size=3, strides=1, padding='same')
        x = tf.nn.relu(x)
        x=tf.keras.layers.GaussianNoise(.2)(x)
        x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding='same')
        x = tf.nn.relu(x)
        x = tf.keras.layers.GaussianNoise(.2)(x)
        x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding='same')
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters=1, kernel_size=3, strides=1, padding='same')
        final = tf.nn.relu(x) #ref1_guess


       # x = tf.contrib.layers.instance_norm(x) #image translation module 2: ref1_guess--->mov1'
        x=tf.concat([final,mov1],axis=3)
        x = tf.layers.conv2d(x, filters=64, kernel_size=6, strides=1, padding='same')
        x = tf.nn.relu(x)
        x = tf.keras.layers.GaussianNoise(.2)(x)
        x = tf.layers.conv2d(x, filters=64, kernel_size=6, strides=1, padding='same')
        x = tf.nn.relu(x)
        x = tf.keras.layers.GaussianNoise(.2)(x)
        x = tf.layers.conv2d(x, filters=64, kernel_size=6, strides=1, padding='same')
        x = tf.nn.relu(x)
        x = tf.keras.layers.GaussianNoise(.2)(x)
        x = tf.layers.conv2d(x, filters=1, kernel_size=6, strides=1, padding='same')
        result_prime= tf.nn.relu(x)


        x=tf.concat([result_prime,mov1],axis=3) #registration module 2:
        x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=1, padding='same')
        x = tf.nn.elu(x)
        # x=tf.contrib.layers.batch_norm(x)
        x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, padding='same')
        x = tf.nn.elu(x)
        # x=tf.contrib.layers.batch_norm(x)
        x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, padding='same')
        x = tf.nn.elu(x)
        # x=tf.contrib.layers.batch_norm(x)
        x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, padding='same')
        x = tf.nn.elu(x)
        # x=tf.contrib.layers.batch_norm(x)
        x = tf.layers.flatten(x)
        # x = tf.layers.dense(x,1024,activation='relu')
        deform1 = tf.layers.dense(x, 3200, activation='linear')
        outsize = (192), (144)
        with tf.Session() as sess:
            with tf.device("/gpu:0"):
                with tf.variable_scope('spatial_transformer'):
                    stl = ET(outsize)
                    stl1=ElasticTransformer(outsize)
                    mov_guess = stl.transform(result_prime, deform1)
                    mov_guess1=stl1.transform(result_prime,-deform)
        # x=tf.contrib.layers.batch_norm(x)

    return final,result_prime,mov_guess,result,deform,deform1,mov_guess1

#data
mov1 = tf.placeholder(tf.float32, shape=[5, 192,144, 1])
ref1 = tf.placeholder(tf.float32, [5, 192,144, 1])
sobel1=tf.placeholder(tf.float32,[5,192,144,1])
sobel2=tf.placeholder(tf.float32,[5,192,144,1])
tot_image1 = tf.placeholder(tf.float32, [20, 48,36, 2])
output1 = tf.placeholder(tf.float32, [20, 1])
tot_image2=tf.placeholder(tf.float32,[20,48,36,2])
output2=tf.placeholder(tf.float32,[20,1])



final,result_prime,mov_guess,result,deform,deform1,mov_guess1=generator(mov1,sobel1,sobel2)
gen_loss=tf.losses.mean_squared_error(mov_guess,mov1)*10000+tf.losses.mean_squared_error(final,ref1)*10000+tf.losses.mean_squared_error(mov_guess,mov_guess1)*10000
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
gen_step=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=gen_loss,var_list=gen_vars)

# sess = tf.Session(config=config)
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)



def fit_gen(batch_x, batch_y,batch_z,batch_a):
    tf.reset_default_graph()
    loss_gen = sess.run([gen_step,gen_loss], {mov1: batch_x, ref1: batch_y,sobel1:batch_z,sobel2:batch_a})
    # accuracy=self.sess.run([self.accuracy],{self.x:batch_x_class,self.y:batch_y_class})
    return loss_gen


def getvalue_gen(batch_x, batch_y,batch_z,batch_a):
    finals,result_init,results,mov_guess1 = sess.run([final,result,result_prime,mov_guess], {mov1: batch_x, ref1: batch_y,sobel1:batch_z,sobel2:batch_a})

    return finals,result_init,results,mov_guess1