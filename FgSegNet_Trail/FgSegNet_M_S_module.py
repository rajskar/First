#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 22:57:12 2017

@author: longang
"""

import keras
from keras.models import Model
from keras.layers import Activation, Input, Dropout, BatchNormalization, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers

from my_upsampling_2d import MyUpSampling2D
import keras.backend as K
import tensorflow as tf

def loss(y_true, y_pred):
    void_label = -1.
    y_pred = K.reshape(y_pred, [-1])
    y_true = K.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def acc(y_true, y_pred):
    void_label = -1.
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def loss2(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def acc2(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


class FgSegNet_M_S_module(object):
    
    def __init__(self, lr, reg, img_shape, scene, vgg_weights_path):
        self.lr = lr
        self.reg = reg
        self.img_shape = img_shape
        self.scene = scene


    def transposedConv(self, x):
        
        # block 5
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='block5_tconv1', 
                                                kernel_regularizer=regularizers.l2(self.reg))(x)     
        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='block5_tconv2')(x)       
        x = Conv2DTranspose(512, (1, 1), activation='relu', padding='same', name='block5_tconv3')(x)
        
        # block 6
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='block6_tconv1', 
                                                kernel_regularizer=regularizers.l2(self.reg))(x)
        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), activation='relu', padding='same', name='block6_tconv2')(x)
        x = Conv2DTranspose(256, (1, 1), activation='relu', padding='same', name='block6_tconv3')(x)
        
        # block 7
        x = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', name='block7_tconv1', 
                                                kernel_regularizer=regularizers.l2(self.reg))(x)
        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='block7_tconv2')(x)
        x = Conv2DTranspose(128, (1, 1), activation='relu', padding='same', name='block7_tconv3')(x)
               
        # block 8
        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), activation='relu', padding='same', name='block8_conv1', 
                                                kernel_regularizer=regularizers.l2(self.reg))(x)
        
        # block 9
        x = Conv2DTranspose(1, (1, 1), padding='same', name='block9_conv1')(x)
        x = Activation('sigmoid')(x)
        
        return x

    def initModel_M(self, dataset_name):
        
        print('New model 4')
        assert dataset_name in ['CDnet', 'SBI', 'UCSD'], 'dataset_name must be either one in ["CDnet", "SBI", "UCSD"]]'
        assert len(self.img_shape)==3
        h, w, d = self.img_shape
        
        input_1 = Input(shape=(h, w, d), name='ip_scale1')        
        # Transposed Conv
        top = self.transposedConv(input_1)
        
        vision_model = Model(inputs= input_1, outputs=top, name='vision_model')
        opt = keras.optimizers.RMSprop(lr = self.lr, rho=0.9, epsilon=1e-08, decay=0.0)

        c_loss = loss
        c_acc = acc
            
        vision_model.compile(loss=c_loss, optimizer=opt, metrics=[c_acc])
        return vision_model
    
    ### FgSegNet_S
    
    def FPM(self, x):
        x1 = MaxPooling2D((2, 2), strides=(1,1), padding='same')(x)
        x1 = Conv2D(64, (1, 1), padding='same')(x1)
        
        x2 = Conv2D(64, (3, 3), padding='same')(x)
        
        x3 = Conv2D(64, (3, 3), padding='same', dilation_rate=4)(x)
        
        x4 = Conv2D(64, (3, 3), padding='same', dilation_rate=8)(x)
        
        x5 = Conv2D(64, (3, 3), padding='same', dilation_rate=16)(x)
        
        x = keras.layers.concatenate([x1, x2, x3, x4, x5], axis=-1)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout2D(0.25)(x)
        return x
        
    def initModel_S(self, dataset_name):
        assert dataset_name in ['CDnet', 'SBI', 'UCSD'], 'dataset_name must be either one in ["CDnet", "SBI", "UCSD"]]'
        assert len(self.img_shape)==3
        h, w, d = self.img_shape
        
        input_1 = Input(shape=(h, w, d), name='input')
        vgg_layer_output = self.VGG16(input_1)
        model = Model(inputs=input_1, outputs=vgg_layer_output, name='model')
        model.load_weights(self.vgg_weights_path, by_name=True)
        
        unfreeze_layers = ['block4_conv1','block4_conv2', 'block4_conv3']
        for layer in model.layers:
            if(layer.name not in unfreeze_layers):
                layer.trainable = False
                
        x = model.output
        
        if dataset_name=='CDnet':
            x1_ups = {'streetCornerAtNight':(0,1), 'tramStation':(1,0), 'turbulence2':(1,0)}
            for key, val in x1_ups.items():
                if self.scene==key:
                    # upscale by adding number of pixels to each dim.
                    x = MyUpSampling2D(size=(1,1), num_pixels=val)(x)
                    break
                
        x = self.FPM(x)
        x = self.transposedConv(x)
        
        if dataset_name=='CDnet':
            if(self.scene=='tramCrossroad_1fps'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0))(x)
            elif(self.scene=='bridgeEntry'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,2))(x)
            elif(self.scene=='fluidHighway'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0))(x)
            elif(self.scene=='streetCornerAtNight'): 
                x = MyUpSampling2D(size=(1,1), num_pixels=(1,0))(x)
                x = Cropping2D(cropping=((0, 0),(0, 1)))(x)
            elif(self.scene=='tramStation'):  
                x = Cropping2D(cropping=((1, 0),(0, 0)))(x)
            elif(self.scene=='twoPositionPTZCam'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(0,2))(x)
            elif(self.scene=='turbulence2'):
                x = Cropping2D(cropping=((1, 0),(0, 0)))(x)
                x = MyUpSampling2D(size=(1,1), num_pixels=(0,1))(x)
            elif(self.scene=='turbulence3'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0))(x)
        
        vision_model = Model(inputs=input_1, outputs=x, name='vision_model')
        opt = keras.optimizers.RMSprop(lr = self.lr, rho=0.9, epsilon= K.epsilon , decay=0.)
        
        # Since UCSD has no void label, we do not need to filter out
        if dataset_name == 'UCSD':
            c_loss = loss2
            c_acc = acc2
        else:
            c_loss = loss
            c_acc = acc
            
        vision_model.compile(loss=c_loss, optimizer=opt, metrics=[c_acc])
        return vision_model