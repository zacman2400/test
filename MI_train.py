import numpy as np
import tensorflow as tf
#import data_processing
#import tensorflow.keras as keras
import PIL
import cv2
import math
from cv2 import remap
import random
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
import os
'''
data = np.zeros([5, 22, 40, 192, 144])
for m in range(0, 5):
    path = 'ALL_Water{}.dat'.format(m + 1)
    myarray = np.fromfile(path, dtype=np.float32)
    myarray = np.reshape(myarray, [22, 176, 144, 192])
    a = np.rot90(myarray, k=3, axes=(3, 2))

    from skimage import exposure

    a = exposure.rescale_intensity(a)
    for k in range(40, 80):
        for i in range(0, 192):
            for j in range(0, 144):
                data[m, :, k - 40, i, j] = np.squeeze(a[:, k, i, j])
data = np.reshape(data, [200,22, 192, 144])
'''

def affine_transform(image, M):
    shape = image.shape
    shape_size = shape[:2]
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    dx_aff = np.zeros([192,144])
    dy_aff = np.zeros([192,144])
    for i in range(0, 192):
        for j in range(0, 144):
            xnew = M[0, 0] * x[i, j] + M[0, 1] * y[i, j] + M[0, 2]
            ynew = M[1, 0] * x[i, j] + M[1, 1] * y[i, j] + M[1, 2]
            dx_aff[i, j] = (xnew - x[i, j])
            dy_aff[i, j] = ynew - y[i, j]
    return dx_aff, dy_aff, x, y


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    new = (144, 192)
    y_final = np.zeros([192,144])
    x_final = np.zeros([192,144])
    # Elastic
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices_final = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    kappa_elastic = map_coordinates(image, indices_final, order=1).reshape(shape)

    # Affine
    tx = random.uniform(-2, 2)
    ty = random.uniform(-4, 4)
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    dx_aff, dy_aff, x_inter, y_inter = affine_transform(kappa_elastic, M)
    indices = np.reshape(y - dy_aff + dy + ty, (-1, 1)), np.reshape(x - dx_aff + dx + tx, (-1, 1))
    kappa = map_coordinates(image, indices, order=1, mode='constant', prefilter=False).reshape(shape)
    return kappa, -dx_aff + dx + tx, -dy_aff + dy + ty, M


def roll(alpha_max, alpha_min, sigma_max, sigma_min, alpha_affine_max, alpha_affine_min):
    a = random.uniform(alpha_min, alpha_max)
    b = random.uniform(sigma_min, sigma_max)
    c = random.uniform(alpha_affine_min, alpha_affine_max)
    return a, b, c


def gamma_correction1(img1, img2):
    rand1 = random.uniform(.1, 2)
    rand2 = random.uniform(.1, 2)
    transformed_bright1 = exposure.adjust_gamma(img1, rand1)
    transformed_bright2 = exposure.adjust_gamma(img2, rand1)
    return transformed_bright1, transformed_bright2

def gamma_correction2(img1, img2):
    rand1 = random.uniform(1,2)
    rand2 = random.uniform(1, 2)
    transformed_bright1 = exposure.adjust_gamma(img1, rand1)
    transformed_bright2 = exposure.adjust_gamma(img2, rand2)
    return transformed_bright1, transformed_bright2

def gamma_correction3(img1, img2):
    rand1 = random.uniform(.1, 2)
    rand2 = random.uniform(2, 2)
    transformed_bright1 = exposure.adjust_gamma(img1, rand1)
    transformed_bright2 = exposure.adjust_gamma(img2, rand2)
    return transformed_bright1, transformed_bright2

def MI_calculation(mov,ref):
    joint, x, y = np.histogram2d(mov.ravel(), ref.ravel(), bins=50)
    pa = np.zeros([50, 50])
    pb = np.zeros([50, 1])
    pc = np.zeros([50, 1])
    mov_single, x = np.histogram(mov.ravel(), bins=50)
    ref_single, x = np.histogram(ref.ravel(), bins=50)
    MI = np.zeros([50, 50])
    for i in range(0, 50):
        for j in range(0, 50):
            pa[i, j] = joint[i, j] / np.sum(joint)
    for i in range(0, 50):
        pb[i, 0] = mov_single[i] / np.sum(mov_single)
        pc[i, 0] = ref_single[i] / np.sum(ref_single)
    for i in range(0, 50):
        for j in range(0, 50):
            if pb[i] > 0 and pc[j] > 0 and pa[i, j] > 0:
                MI[i, j] = pa[i, j] * np.log(pa[i, j] / (pb[i] * pc[j]))
            else:
                MI[i, j] = 0

    MI_tot = np.sum(np.sum(MI))
    return MI_tot

def transform_dis1(select,init, batch_size, l):
    x_train = np.zeros([batch_size * 2,48,36, 1])
    y_train = np.zeros([batch_size * 2, 1])
    x_final = np.zeros([batch_size * 2, 48,36, 2])
    y_final = np.zeros([batch_size * 2, 1])
    x_shuffle = np.zeros([batch_size, 96, 72, 2])
    y_shuffle = np.zeros([batch_size, 1])
    batch = batch_size
    batch = np.array(batch)
    mean = 0.0  # some constant
    std = 0  # some constant (standard deviation)
    # y_train=np.zeros(5000,2,3,1)
    alpha = 0
    beta = 0
    gamma = 0

    for i in range(0, batch_size):
        a, b, c = roll(120, 70, 25, 12, 2, -2)
        # for j in range(0,5):
        #  im1,im2=gamma_correction(data[l[i],:,:],data[l[i],:,:])
        # ima,imb=contrast(im1,im2)
        # if alpha<75:
        #     y_train[alpha]=1
        #    x_train[alpha,:,:,0]=data[l[i],:,:]
        #    x_train[alpha,:,:,1]=im2
        # else:


        warp, m, n, M = elastic_transform(select[i,:,:], a, b, c)
       # mov, ref = gamma_correction2(warp, data[l[i], :, :])
        x_train[alpha, :, :, 0] =
        x_train[alpha,:,:,1]=select[i,:,:]
        y_train[alpha, :] = MI_calculation(warp,select[i,:,:])

        alpha = alpha + 1
    # affine_train[alpha,:,:,0]=Q
    # affine_train[alpha,:,:,1]=cv2.invertAffineTransform(Q,A)
    # n=np.random.randint(low=0,high=batch_size*5,size=batch_size*5)
    # x_shuffle=x_train[n,:,:,:]
    # y_shuffle=y_train[n]
    m = np.random.randint(low=0, high=batch_size, size=batch_size)
    x_final = x_train[m, :, :, :]
    y_final = y_train[m, :]
    return x_final, y_final
'''
def transform_dis2(select, batch_size, l):
    x_train = np.zeros([batch_size * 2, 192, 144, 2])
    y_train = np.zeros([batch_size * 2, 1])
    x_final = np.zeros([batch_size * 2, 192, 144, 2])
    y_final = np.zeros([batch_size * 2, 1])
    x_shuffle = np.zeros([batch_size, 96, 72, 2])
    y_shuffle = np.zeros([batch_size, 1])
    batch = batch_size
    batch = np.array(batch)
    mean = 0.0  # some constant
    std = 0  # some constant (standard deviation)
    # y_train=np.zeros(5000,2,3,1)
    alpha = 0
    beta = 0
    gamma = 0

    for i in range(0, 2 * batch_size):
        a, b, c = roll(120, 70, 25, 12, 2, -2)
        # for j in range(0,5):
        #  im1,im2=gamma_correction(data[l[i],:,:],data[l[i],:,:])
        # ima,imb=contrast(im1,im2)
        # if alpha<75:
        #     y_train[alpha]=1
        #    x_train[alpha,:,:,0]=data[l[i],:,:]
        #    x_train[alpha,:,:,1]=im2
        # else:

        if alpha < batch_size:
            warp, m, n, M = elastic_transform(data[l[i], :, :], a, b, c)
            mov, ref = gamma_correction2(warp, data[l[i], :, :])
            x_train[alpha, :, :, 0] = mov
            x_train[alpha,:,:,1]=ref
            y_train[alpha, :] = 1
        else:
            warp, m, n, M = elastic_transform(data[l[i-10], :, :], a, b, c)
            mov, ref = gamma_correction3(warp, data[l[i - 10], :, :])
            x_train[alpha, :, :, 0] = mov
            x_train[alpha,:,:,1]=ref
            y_train[alpha, :] = 0
            # y_train[alpha]=0
        alpha = alpha + 1
    # affine_train[alpha,:,:,0]=Q
    # affine_train[alpha,:,:,1]=cv2.invertAffineTransform(Q,A)
    # n=np.random.randint(low=0,high=batch_size*5,size=batch_size*5)
    # x_shuffle=x_train[n,:,:,:]
    # y_shuffle=y_train[n]
    m = np.random.randint(low=0, high=2 * batch_size, size=2 * batch_size)
    x_final = x_train[m, :, :, :]
    y_final = y_train[m, :]
    return x_final, y_final
'''

def transform_gen(select, init, batch_size, l,f):
    x_train_norm = np.zeros([batch_size, 192,144, 2])
    y_final = np.zeros([batch_size * 2, 1])
    x_shuffle = np.zeros([batch_size, 192,144, 2])
    y_shuffle = np.zeros([batch_size, 1])
    batch = batch_size
    batch = np.array(batch)
    mean = 0.0  # some constant
    std = 0  # some constant (standard deviation)
    # y_train=np.zeros(5000,2,3,1)
    alpha = 0
    beta = 0
    gamma = 0

    for i in range(0, batch_size):
        a, b, c = roll(120, 70, 15, 10, 2, -2)
        kappa=f[i]
        zed=l[i]
        warp,m,n,M= elastic_transform(select[i,:,:], a, b, c)
        #mov, ref = gamma_correction2(warp, data[l[i], :, :])
        x_train_norm[alpha, :, :, 0] = warp
        x_train_norm[alpha, :, :, 1] = init[i,:,:]
        alpha = alpha + 1
    # affine_train[alpha,:,:,0]=Q
    # affine_train[alpha,:,:,1]=cv2.invertAffineTransform(Q,A)
    # n=np.random.randint(low=0,high=batch_size*5,size=batch_size*5)
    # x_shuffle=x_train[n,:,:,:]
    # y_shuffle=y_train[n]

    return x_train_norm