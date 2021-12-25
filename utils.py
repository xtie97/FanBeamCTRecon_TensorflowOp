import tensorflow as tf 
import numpy as np 
import math
import matplotlib.pylab as plt 
from tensorflow.python.framework import ops
from tensorflow.keras import models
from tensorflow.keras import layers 
from tensorflow.python.framework.tensor_util import make_tensor_proto
from config import config
FP_module = tf.load_op_library('./FP/FP.so')
BP_module = tf.load_op_library('./BP/BP.so')
FBP_module = tf.load_op_library('./FBP/FBP.so')

##################################################################
def ForwardProjection(img, nimg=1):
    '''
    nimg: # output (if water+iodine, nimg=2)
    '''
    nview = config.ProjectionsPerRotation
    nCol = config.DetectorColumnNumber
    fstartangle = config.StartAngle
    nX, nY = config.NX, config.NY
    angle_proto = angle(nview, fstartangle)
    para_proto = param_set(nimg)
    sino = FP_module.fp_fan_curve(volume = img, angle = angle_proto, parameter = para_proto, vol_shape = [nimg, nX, nY], proj_shape = [nimg, nview, nCol])
    return sino 

def BackProjection(sino, nimg=1):
    nview = config.ProjectionsPerRotation
    nCol = config.DetectorColumnNumber
    fstartangle = config.StartAngle
    nX, nY = config.NX, config.NY
    angle_proto = angle(nview, fstartangle)
    para_proto = param_set(nimg)
    img = BP_module.bp_fan_curve(projection = sino, angle = angle_proto, parameter = para_proto, vol_shape = [nimg, nX, nY], proj_shape = [nimg, nview, nCol])
    return img 

def FilteredBackProjection(sino, nimg=1):
    nview = config.ProjectionsPerRotation
    nCol = config.DetectorColumnNumber
    fstartangle = config.StartAngle
    nX, nY = config.NX, config.NY
    angle_proto = angle(nview, fstartangle)
    para_proto = param_set(nimg)
    img = FBP_module.fbp_fan_curve(projection = sino, angle = angle_proto, parameter = para_proto, vol_shape = [nimg, nX, nY], proj_shape = [nimg, nview, nCol])
    return img
    
##################################################################    
def angle(nview, fstartangle):
    dAngle = 2.0*math.pi/nview
    anglePos = np.arange(nview, dtype=np.float32)
    for i in range(nview):
        anglePos[i] = fstartangle + i*dAngle
    return make_tensor_proto(anglePos, tf.float32)
        
def param_set(nimg):
    sdd = config.SourceDetectorDistance
    sod = config.SourceAxisDistance
    nfan = config.FanAngle
    nCol = config.DetectorColumnNumber 
    dcol = nfan/nCol
    dshift = config.DetectorShift
    dLeft = -nfan/2 - dshift*dcol # angle (degree) 
    ############################
    nview = config.ProjectionsPerRotation 
    rfov = config.FOVRadius  
    ###########################
    nrange = config.FullAngle
    fstartangle = config.StartAngle
    Batch_size = 1
        
    ############################
    parameters = np.arange(16, dtype=np.float32)
    parameters[0] = sdd
    parameters[1] = sod
    parameters[2] = nview
    parameters[3] = nrange
    parameters[4] = nCol
    parameters[5] = config.NX
    parameters[6] = config.NY
    parameters[7] = rfov
    parameters[8] = dcol
    parameters[9] = dshift  
    parameters[10] = dLeft 
    parameters[11] = Batch_size*nimg
    parameters[12] = fstartangle # start angle
     
    return make_tensor_proto(parameters, tf.float32)
        
