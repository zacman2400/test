#import image_trans as transform
import tensorflow as tf
import transformed as transform
import hist as h
from config2 import get_config
from ops import mkdir
import tensorflow as tf
import data_proc
import data_proc_1
import data_proc
import tensorflow.keras as keras
import PIL
import cv2
import math
import numpy as np


from cv2 import remap
import random
import matplotlib
from matplotlib import pyplot as plt
#from PIL import Image, ImageEnhance

#data loading
data = np.zeros([5, 22, 40, 192,144])
m_data = np.zeros([200, 22,192,144])
alpha=0
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
for i in range(0,5):
    for j in range(0,40):
        m_data[alpha,:,:,:]=data[i,:,j,:,:]
        alpha=alpha+1

import data_proc_19
import time
import numpy
import scipy
from scipy import ndimage

tf.reset_default_graph()
batch_xp = np.zeros([2, 10, 192, 144])
batch_yp = np.zeros([2, 10, 192, 144])
batch_x_grad = np.zeros([10, 192, 144])
batch_y_grad = np.zeros([10, 192, 144])
mov = np.zeros([22, 192 * 144])
ref = np.zeros([22, 192, 144])
mov_hist = np.zeros([22, 100])
ref_hist = np.zeros([22, 100])
tot = np.zeros([5, 192, 144])
tot1 = np.zeros([5, 192, 144])

#train network
for i in range(0, 5000):
    for j in range(0, 5000):
        batch_size = 5
        l = np.random.randint(low=0, high=10, size=batch_size)
        m = np.random.randint(low=20, high=30, size=batch_size)
        p = np.random.randint(low=0, high=22, size=batch_size)
        x_select, x_tester = data_proc_19.transform_gen(m_data[m, l, :, :], m_data[m, 21, :, :], batch_size, l, m)
        for k in range(0, batch_size):
            sobelx = cv2.Sobel(x_select[k, :, :, 1], cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(x_select[k, :, :, 1], cv2.CV_64F, 0, 1, ksize=5)
            tot[k, :, :] = np.hypot(sobelx, sobely)
            sobelx1 = cv2.Sobel(x_select[k, :, :, 0], cv2.CV_64F, 1, 0, ksize=5)
            sobely1 = cv2.Sobel(x_select[k, :, :, 0], cv2.CV_64F, 0, 1, ksize=5)
            tot[k, :, :] = np.hypot(sobelx, sobely)
            tot1[k, :, :] = np.hypot(sobelx1, sobely1)

        tot_ref = np.expand_dims(tot, axis=3)
        tot_mov = np.expand_dims(tot1, axis=3)
        x_s1 = np.expand_dims(x_select[:, :, :, 0], axis=3)
        x_s2 = np.expand_dims(x_select[:, :, :, 1], axis=3)
        batch_x = x_s1
        batch_y = x_s2
        batch_z = tot_ref
        batch_a = tot_mov
        loss = transform.fit_gen(batch_x, batch_y, batch_z)
        finals, results, results_prime, mov_guess1 = transform.getvalue_gen(batch_x, batch_y, batch_z)
        final = np.array(finals)
        result_init = np.array(results)
        mov_guess = np.array(mov_guess1)

        # orig_guess=np.array(orig_guess)

        # deform=np.array(deform)
        # transform.save(batch_x)

        # result=np.array(result)
        print(loss)
    # time.sleep(5)