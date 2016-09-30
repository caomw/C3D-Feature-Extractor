# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 11:09:38 2016

@author: jimmy

"""

import numpy as np
import cv2
from caffe_io import transform_image, load_image


# input a list of normalized images  SequenceID x 1 x Channel x Height x Width
# output a volume of 1 x Channle X SequenceID X Height X Width, sequenceID is sampled in temporal space
# 1 x 3 x 16 x 112 x 112 for C3D network input
def image_sequence_transform(images):
    assert(len(images) == 16)
    data = np.empty((1, 3, 16, 112, 112))
    
    for i in range(len(images)):
        for j in range(3):
            data[0, j, i, : ] = images[i][0, j, :]
    return data

# folder /ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01
# start_frame_num 1
def load_image_sequence(folder, start_frame_num, frame_num = 16):
    imgs = []
    for i in range(start_frame_num, start_frame_num + frame_num):
        name = folder + '/%08d.jpg' % (i)
        img = load_image(name)
        imgs.append(img)
    return imgs         

if __name__ == '__main__':
    folder = '/Users/jimmy/Desktop/images/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01'
    imgs = [];
    for i in range(1, 17):
        img_name = folder + '/image_%04d.jpg' % i
        img = load_image(img_name)
        imgs.append(img)

    transformed_imgs = []
    for img in imgs:
        data = transform_image(img, over_sample = False, mean_pix = [103.939, 116.779, 123.68], image_dim = 120, crop_dim = 112)
        transformed_imgs.append(data)
    
    c3d_input = image_sequence_transform(transformed_imgs)
            
    
    
