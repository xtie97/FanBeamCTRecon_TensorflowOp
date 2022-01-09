import tensorflow as tf 
import numpy as np
import matplotlib.pylab as plt
import math
import psutil

FBP_module = tf.load_op_library('./FBP_op.so')

def read_binary_dataset(file_name, num_samples, nview, ncol):   # please pay attentions to the order of data!
    
    f = open(file_name, 'rb')
    proj = np.fromfile(f, dtype=np.float32, count=-1)
    f.close()
    proj = np.reshape(proj, (num_samples, nview, ncol), order='C')

    return proj

def FDK_generate(index, fstart_angle, nview_full, nslice, folder='Training'):
    # Glabol variables:
    #num_gpus = 1
    sdd = 946.73233346
    sod = 538.52
    ncol = 888
    nfan = 55.0059
    dcol = nfan/ncol
    nimg = 512
    dz = 2.5
    shift = 1.125
    dLeft = -27.50295
    #beta = nfan+180
    det_bottom = -35.16105
    drow = 1.0987828125 
    ############################
    #index = 1
    # fstart_angle = 56.4605480091944 # 56.46054 
    # nview_full = 1686 # need change
    # nslice = 109 
    # folder = 'Training'
    #################################
    nview = nview_full
    fstart_angle = -fstart_angle 
    prj_shift = int(round(fstart_angle/(360/nview)))
    #print(prj_shift)

    exam_name = 'exam' + str(index)
    proj_high = read_binary_dataset('/home/xintie/DSI_Project/DSI_data_sharing/' + folder + '/' + exam_name + '_high.prj', nslice, nview, ncol) # 888 1686 
    proj_m1 = read_binary_dataset('/home/xintie/DSI_Project/DSI_data_sharing/' + folder + '/' + exam_name + '_m1.prj', nslice, nview, ncol) # 888 1686 
    proj_m2 = read_binary_dataset('/home/xintie/DSI_Project/DSI_data_sharing/' + folder + '/' + exam_name + '_m2.prj', nslice, nview, ncol) # 888 1686 
    proj_eff = read_binary_dataset('/home/xintie/DSI_Project/DSI_data_sharing/'+ folder + '/' + exam_name + '_energy.prj', nslice, nview, ncol) # 888 1686 
    proj_mask = read_binary_dataset('/home/xintie/DSI_Project/DSI_data_sharing/' + folder + '/' + exam_name + '_mask.prj', nslice, nview, ncol)

    rfov = 250
    #sfov = rfov
    #dx = 2*rfov/nimg # 0.7812
    ############################
    nrange = nview*360/nview_full
    nebin = 2
    Batch_size = 1

    ############################
    parameters = np.arange(16, dtype=np.float32)
    parameters[0] = sdd
    parameters[1] = sod
    parameters[2] = nview
    parameters[3] = nrange
    parameters[4] = ncol
    parameters[5] = nimg
    parameters[6] = nimg
    parameters[7] = rfov
    parameters[8] = dcol
    parameters[9] = shift
    parameters[10] = dLeft 
    parameters[11] = Batch_size
    parameters[12] = 0  #fstart_angle=150 # start angle
    parameters[13] = det_bottom # Detector bottom
    parameters[14] = drow 
    parameters[15] = dz
    para_proto = tf.contrib.util.make_tensor_proto(parameters, tf.float32)
    #dAngle = 360.0/nview*math.pi/180.0
    dAngle = 2.0*math.pi/nview
    anglePos = np.arange(nview, dtype=np.float32)
    for i in range(nview):
        anglePos[i] = fstart_angle + i*dAngle
            
    angle_proto = tf.contrib.util.make_tensor_proto(anglePos, tf.float32)
    ##########################################
    print(prj_shift)
    new_proj_high = np.concatenate((proj_high[:, prj_shift: proj_high.shape[1], :], proj_high[:, 0:prj_shift, :]), axis = 1)
    new_proj_m1 = np.concatenate((proj_m1[:, prj_shift: proj_m1.shape[1], :], proj_m1[:, 0:prj_shift, :]), axis = 1)
    new_proj_m2 = np.concatenate((proj_m2[:, prj_shift: proj_m2.shape[1], :], proj_m2[:, 0:prj_shift, :]), axis = 1)
    new_proj_eff = np.concatenate((proj_eff[:, prj_shift: proj_eff.shape[1], :], proj_eff[:, 0:prj_shift, :]), axis = 1)
    new_proj_mask = np.concatenate((proj_mask[:, prj_shift: proj_mask.shape[1], :], proj_mask[:, 0:prj_shift, :]), axis = 1)
    
    del proj_high, proj_m1, proj_m2, proj_eff, proj_mask

    savepath = '/home/xintie/DSI_Project/Data_npy/' + folder + '/'
    
    index_data = []

    for i in range(nslice):
        a = new_proj_high[i,:,:].reshape(1, nview, ncol)/49000
        b = new_proj_m1[i,:,:].reshape(1, nview, ncol)
        c = new_proj_m2[i,:,:].reshape(1, nview, ncol)
        d = new_proj_eff[i,:,:].reshape(1, nview, ncol)
        e = new_proj_mask[i,:,:].reshape(1, nview, ncol)
        
        a_ = tf.convert_to_tensor(a, dtype=tf.float32) #a_ = tf.reshape(a_, [1, nview, ncol], name=None)
        b_ = tf.convert_to_tensor(b, dtype=tf.float32)
        c_ = tf.convert_to_tensor(c, dtype=tf.float32)

        a_ = FBP_module.BackProjection(projection=a_, angle = angle_proto, parameter=para_proto, vol_shape=[1,nimg,nimg], proj_shape=[1, nview, ncol])
        b_ = FBP_module.BackProjection(projection=b_, angle = angle_proto, parameter=para_proto, vol_shape=[1,nimg,nimg], proj_shape=[1, nview, ncol])
        c_ = FBP_module.BackProjection(projection=c_, angle = angle_proto, parameter=para_proto, vol_shape=[1,nimg,nimg], proj_shape=[1, nview, ncol])
        with tf.Session(''):
            a_ = a_.eval()
            b_ = b_.eval()
            c_ = c_.eval()
        
        index_data_ = str(index) + '_' + str(i+1) 
        index_data.append(index_data_)
        
        np.save(savepath + 'high_prj_' + index_data_ + '.npy', a.reshape(1, nview*ncol, 1)) # mu is the unit 
        np.save(savepath + 'm1_prj_' + index_data_ + '.npy', b.reshape(1, nview*ncol, 1))
        np.save(savepath + 'm2_prj_' + index_data_ + '.npy', c.reshape(1, nview*ncol, 1))
        np.save(savepath + 'eff_prj_' + index_data_ + '.npy', d.reshape(1, nview*ncol, 1))
        np.save(savepath + 'mask_prj_' + index_data_ + '.npy', e.reshape(1, nview*ncol, 1))

        np.save(savepath + 'high_rcn_' + index_data_ + '.npy', a_.reshape(1, nimg, nimg, 1))
        b_ = b_.reshape(1, nimg, nimg, 1)
        c_ = c_.reshape(1, nimg, nimg, 1)
        m_ = np.concatenate((b_, c_), axis=-1)
        np.save(savepath + 'm_rcn_' + index_data_ + '.npy', m_)
        
        f=open('./recon_images/recon_' + str(i)  + '.raw', "wb")
        arr=bytearray(a_)
        f.write(arr)
        f.close()
        del arr    
         
        del a, b, c, d, e
        del a_, b_, c_, m_
    del new_proj_high, new_proj_m1, new_proj_m2, new_proj_eff, new_proj_mask
    
    #image_list_file  = savepath + 'read_list_training_loss.txt'
    #if index == 1:
    #    with open(image_list_file, "w") as txt_file:
    #        for listitem in index_data:
    #            txt_file.write('%s\n' % listitem)
    #else:
    #    with open(image_list_file, "a") as txt_file:
    #        for listitem in index_data:
    #            txt_file.write('%s\n' % listitem)

    

if __name__ == "__main__":  
     
    #index_list = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,\
    #    23, 25, 26, 27, 28, 29, 31, 32]
    #index_list = [35, 36, 38, 39, 45, 47, 48, 51, 52, 54]
    index_list = [42]
    
    #fstart_angle_list = [56.4605480091944, 136.528068932073, -127.70155229882, -145.194984989732, 110.124579336798,\
    #    -92.6097786264378, 145.323959224458, -158.882716529836, 23.8351249102531, -28.7200525263552,\
    #        32.6819410930904, 21.479406325894, -4.58546981108839, 83.2733025395366, 98.6784936841769, \
    #            -12.6559257117271, 137.334468386241, 92.7357100118889, -48.5079028538258, 150.793465428778,\
    #                83.1901654769592, 119.592314356703, -38.082247463205, 0.872302128237038, 102.461107088043,\
    #                    -91.4014635207493, -39.012327982864, -10.6075383192619, -23.5882991764928]
    #fstart_angle_list = ((np.array([5.2223, 3.1231, 1.5537, 1.5329, 1.6248, 3.0283, 5.9339, 1.5710, 4.8058, 3.2332]))-math.pi)/math.pi*180
    #print(fstart_angle_list[0])
    fstart_angle_list = [0.0] 
    
    #nview_list = [1686, 1968, 1968, 1968, 1686, 1968, 1968, 1968, 1968, 1968, 1968, 1968, 1968, 1968, 1686,\
    #    1968, 1722, 1686, 1686, 1968, 1968, 1968, 1968, 1968, 1968, 1968, 1968, 1968, 1968]
    #nview_list = [1968, 1968, 1968, 1968, 1968, 1968, 1968, 1968, 1968, 1968]
    nview_list = [ 1968 ]
    
    #nslice_list = [109, 127, 92, 109, 118, 144, 107, 106, 117, 117, 106, 111, 107, 96, 114, 115, 117, 124,\
    #    129, 105, 108, 99, 111, 117, 120, 108, 115, 109, 105]
    #nslice_list = [99, 96, 115, 110, 126, 138, 135, 125, 128, 221]
    nslice_list = [147]
    
    for ij in range(1): 
        ij = ij + 0 
        FDK_generate(index_list[ij], fstart_angle_list[ij], nview_list[ij], nslice_list[ij], folder='Training')
        
        info = psutil.virtual_memory()
        print('Memory Usage:', info.percent, '%')
        if info.percent > 90:
            break
        
    

















    


    
