#! /usr/bin/python
# -*- coding: utf8 -*-
import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from utils import ForwardProjection, BackProjection, FilteredBackProjection

def read_binary_dataset(file_name, dim):   # please pay attentions to the order of data!\n"
    f = open(file_name, 'rb')
    data = np.fromfile(f, dtype=np.float32, count=-1)
    f.close()
    data = np.reshape(data, (-1, dim[0], dim[1]), order='C') # -1: slice 
    return data

def test():
    img = read_binary_dataset('./test.fbp', [512, 512])
    nimg = img.shape[0]
    ############### FP ###############
    sino = ForwardProjection(img, nimg)
    with tf.Session(''):
        sino = sino.eval()
    f=open('./test_prj.raw', "wb")
    f.write(bytearray(sino))
    f.close()
    
    ############### BP ###############
    img = BackProjection(sino, nimg)
    with tf.Session(''):
        img = img.eval()
    f=open('./test_img_BP.raw', "wb")
    f.write(bytearray(img))
    f.close()
    
    ############### FBP ###############
    img = FilteredBackProjection(sino, nimg)
    with tf.Session(''):
        img = img.eval()
    f=open('./test_img_FBP.raw', "wb")
    f.write(bytearray(img))
    f.close()
    
    
if __name__ == '__main__':
   test()


    


