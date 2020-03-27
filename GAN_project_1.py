# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 23:29:34 2020

@author: sharm
"""
from __future__ import print_function, division
import os
import numpy as np
from glob import glob
import scipy
import matplotlib.pyplot as plt         
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Input, LeakyReLU, UpSampling2D, Conv2D, Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import Model
from keras.optimizers import Adam


class ImageHelper(object):
    def save_image(self,plot_image,epoch):
        os.makedirs('cyclic_gan_images',exist_ok=True)
        titles = ['Original','Transformed']
        fig, axs = plt.subplots(2,2)
        cnt = 0
        for i in range(2):
            for j in range(3):
                axs[i,j].imshow(plot_images[cnt])
                axs[i,j].set_title(title[j])
                axs[i,j].axis('off')
                cnt +=1
                
            fig.savefig("cyclic_gan_images/{}".format(epoch))
            plt.close()
            
        def plot20(self,images_paths_array):
            plt.figure(figsize=(10,8))
            for i in range(20):
                img = plt.imread(images_path_array[i])
                plt.subplot(4,5,i+1)
                plt.imshow(img)
                plt.title(img.shape)
                plt.xticks([])
                plt.yticks([])
                
            plt.tight_layout()
            plt.show()
            
        def load_image(self, path):
            return scipy.misc.imread(path, mode= 'RGB').astype(np.float)
        
        def load_testing_image(self, path):
            self.img_res = (128,128)
            path_X = glob(path + "/testA/*.jpg")
            path_Y = glob(path + "/testB/*.jpg")
            
            image_X = np.random.choice(path_X,1)
            image_Y = np.random.choice(path_Y,1)
            
            img_X = self.load_image(image_X[0])
            img_X = scipy.misc.imresize(img_X, self.img_res)
            if np.random.random() > 0.5:
                img_X = np.fliplr(img_X)
            img_X = np.array(img_X)/127.5 - 1
            img_X = np.expand_dims(img_X, axis =0)
            
            img_Y = self.load_image(image_Y[0])
            img_Y = scipy.misc.imresize(img_Y, self.img_res)
            if np.random.random() > 0.5:
                img_Y = np.fliplr(img_Y)
            img_Y = np.array(img_Y)/127.5 - 1
            img_Y = np.expand_dims(img_Y, axis =0)
            
            return(img_X, img_Y)
            
        def load_batch_of_train_images(self, path, batch_size =1):
            self.img_res = (128,128)
            path_X = glob(path + "/trainA/*.jpg")
            path_Y = glob(path + "/trainB/*.jpg")
            
            
            self.n_batches = int(min(len(path_X),len(path_Y))/batch_size)
            total_samples = self.n_batches*batch_size
            
            path_X = np.random.choice(path_X, total_samples, replace =False)
            path_Y = np.random.choice(path_Y, total_samples, replace =False)
            
            
            for i in range(self.n_batches-1):
                batch_A = path_X[i*batch_size:(i+1)*batch_size]
                batch_B = path_X[i*batch_size:(i+1)*batch_size]
                
                imgs_A,imgs_B = [],[]
            
                for img_A, img_B in zip(batch_A,batch_B):
                    img_A = self.load_image(img_A)
                    img_B = self.load_image(img_B)
                    
                    img_A = scipy.misc.imresize(img_A, self.img_res)
                    img_B = scipy.misc.imresize(img_B, self.img_res)
                    
                    imgs_A.append(img_A)
                    imgs_B.append(img_B)
                    
                imgs_A = np.array(imgs_A)/127.5 - 1
                imgs_B = np.array(imgs_B)/127.5 - 1
                
                yield imgs_A,imgs_B
            

class CycleGAN():
    
    def __init__(self,image_shape,cycle_lambda,image_helper):
        self.optimier = Adam(0.0002, 0.5)
        
        self.cycle_lambda = cycle_lambda
        self.id_lambda = 0.1*self.cycle_lambda
        self._image_helper = image_helper
        self.img_shape = image_shape
        
        patch = int(self.img_shape[0]/2**4)
        self.disc_patch = (patch, patch, 1)
        
        print("Build Discriminators...")
        self._discriminatorX = self._build_discriminator_model()
        self._compile_discriminator_model(self._discriminatorX)
        self._discriminatorY = self._build_discriminator_model()
        self._compile_discriminator_model(self._discriminatorY)
        
        print("Build Generators...")
        self._generatorXY = self._build_generator_model()
        self._generatorYX = self._build_generator_model() 
        
        print("Build GAN...")
        self._build_and_compile_gan()
        
        
    def train(self, epochs,batch_size,train_data_path):
        real = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        
        history =[]
        
        
       
        for epoch in range(epochs):
            for i, (imagesX, imagesY) in enumerate(self._image_helper.load_batch_of_train_images(train_data_path, batch_size)):
                print("---------------------------------------------------------")
                print("******************Epoch {} | Batch {}***************************".format(epoch, i))
                print("Generate images...")
                fakeY = self._generatorXY.predict(imagesX)
                fakeX = self._generatorYX.predict(imagesY)         
                
                
                print("Train Discriminators...")
                discriminatorX_loss_real = self._discriminatorX.train_on_batch(imagesX, real)
                discriminatorX_loss_fake = self._discriminatorX.train_on_batch(fakeX, fake)
                discriminatorX_loss = 0.5 * np.add(discriminatorX_loss_real, discriminatorX_loss_fake)
                
        
                discriminatorY_loss_real = self._discriminatorY.train_on_batch(imagesY, real)
                discriminatorY_loss_fake = self._discriminatorY.train_on_batch(fakeY, fake)
                discriminatorY_loss = 0.5 * np.add(discriminatorY_loss_real, discriminatorY_loss_fake)
                
                mean_discriminator_loss = 0.5 * np.add(discriminatorX_loss, discriminatorY_loss)
                
                print("Train Generators...")
                generator_loss = self.gan.train_on_batch([imagesX, imagesY],
                                                        [real, real,
                                                        imagesX, imagesY,
                                                        imagesX, imagesY])
    
        