from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, UpSampling2D, Reshape, Flatten, LSTM
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.constraints import min_max_norm, non_neg, binary_m, two_value
from PIL import Image
import os
import image_evaluate as ievalue
import time
import method_module as mm
import math

dataset = ['stl','face']
dataset_cifar = ['cifar']
size = np.zeros(3)
size[0] = 32
size[1] = 64
size[2] = 128
size = np.int_(size)
size_cifar = np.zeros(1)
size_cifar[0] = 32
size_cifar = np.int_(size_cifar)
rate = np.arange(10)*0.1-0.1
rate[0] = 0.001
rate[1] = 0.01
print(rate)
mf1 = np.zeros((3,3,10,4))
mf2 = np.zeros((3,3,10,4))
mf3 = np.zeros((3,3,10,4))
mf4 = np.zeros((3,3,10,4))
mf5 = np.zeros((3,3,10,4))
mf6 = np.zeros((3,3,10,4))
mf7 = np.zeros((3,3,10,4))

cmf1 = np.zeros((1,1,10,4))
cmf2 = np.zeros((1,1,10,4))
cmf3 = np.zeros((1,1,10,4))
cmf4 = np.zeros((1,1,10,4))
cmf5 = np.zeros((1,1,10,4))
cmf6 = np.zeros((1,1,10,4))
cmf7 = np.zeros((1,1,10,4))
for i in range(len(dataset_cifar)):
    for j in range(len(size_cifar)):
        for k in range(len(rate)):
            try:
                mse,psnr,ssim,retime = mm.f1(size_cifar[j],rate[k],dataset_cifar[i],0.1,20,64)
                cmf1[i][j][k][0],cmf1[i][j][k][1],cmf1[i][j][k][2],cmf1[i][j][k][3] = mse,psnr,ssim,retime
                print('OK',size_cifar[j],rate[k],dataset_cifar[i],i,j,k)
            except:
                print('error',size_cifar[j],rate[k],dataset_cifar[i],i,j,k)
            try:
                mse,psnr,ssim,retime = mm.f2(size_cifar[j],rate[k],dataset_cifar[i],'PartHadamard',0.1,20,64)
                cmf2[i][j][k][0],cmf2[i][j][k][1],cmf2[i][j][k][2],cmf2[i][j][k][3] = mse,psnr,ssim,retime
                print('OK',size_cifar[j],rate[k],dataset_cifar[i],i,j,k)
            except:
                print('error',size_cifar[j],rate[k],dataset_cifar[i],i,j,k)
            try:
                mse,psnr,ssim,retime = mm.f3(size_cifar[j],rate[k],dataset_cifar[i],0.1,20,64)
                cmf3[i][j][k][0],cmf3[i][j][k][1],cmf3[i][j][k][2],cmf3[i][j][k][3] = mse,psnr,ssim,retime
                print('OK',size_cifar[j],rate[k],dataset_cifar[i],i,j,k)
            except:
                print('error',size_cifar[j],rate[k],dataset_cifar[i],i,j,k)
            try:
                mse,psnr,ssim,retime = mm.f4(size_cifar[j],rate[k],dataset_cifar[i],0.1,20,64)
                cmf4[i][j][k][0],cmf4[i][j][k][1],cmf4[i][j][k][2],cmf4[i][j][k][3] = mse,psnr,ssim,retime
                print('OK',size_cifar[j],rate[k],dataset_cifar[i],i,j,k)
            except:
                print('error',size_cifar[j],rate[k],dataset_cifar[i],i,j,k)
            try:
                mse,psnr,ssim,retime = mm.f5(size_cifar[j],rate[k],dataset_cifar[i],0.1,20,64)
                cmf5[i][j][k][0],cmf5[i][j][k][1],cmf5[i][j][k][2],cmf5[i][j][k][3] = mse,psnr,ssim,retime
                print('OK',size_cifar[j],rate[k],dataset_cifar[i],i,j,k)
            except:
                print('error',size_cifar[j],rate[k],dataset_cifar[i],i,j,k)
            try:
                mse,psnr,ssim,retime = mm.f6(size_cifar[j],rate[k],dataset_cifar[i],0.1,20,64)
                cmf6[i][j][k][0],cmf6[i][j][k][1],cmf6[i][j][k][2],cmf6[i][j][k][3] = mse,psnr,ssim,retime
                print('OK',size_cifar[j],rate[k],dataset_cifar[i],i,j,k)
            except:
                print('error',size_cifar[j],rate[k],dataset_cifar[i],i,j,k)
            try:
                mse,psnr,ssim,retime = mm.f7(size_cifar[j],rate[k],dataset_cifar[i],0.1,20,64)
                cmf7[i][j][k][0],cmf7[i][j][k][1],cmf7[i][j][k][2],cmf7[i][j][k][3] = mse,psnr,ssim,retime
                print('OK',size_cifar[j],rate[k],dataset_cifar[i],i,j,k)
            except:
                print('error',size_cifar[j],rate[k],dataset_cifar[i],i,j,k)
np.save('cmf1.npy',cmf1)
np.save('cmf2.npy',cmf2)
np.save('cmf3.npy',cmf3)
np.save('cmf4.npy',cmf4)
np.save('cmf5.npy',cmf5)
np.save('cmf6.npy',cmf6)
np.save('cmf7.npy',cmf7)
for i in range(len(dataset)):
    for j in range(len(size)):
        for k in range(len(rate)):
            try:
                mse,psnr,ssim,retime = mm.f1(size[j],rate[k],dataset[i],0.1,20,64)
                mf1[i][j][k][0],mf1[i][j][k][1],mf1[i][j][k][2],mf1[i][j][k][3] = mse,psnr,ssim,retime
                print('OK',size[j],rate[k],dataset[i],i,j,k)
            except:
                print('error',size[j],rate[k],dataset[i],i,j,k)
            try:
                mse,psnr,ssim,retime = mm.f2(size[j],rate[k],dataset[i],'PartHadamard',0.1,20,64)
                mf2[i][j][k][0],mf2[i][j][k][1],mf2[i][j][k][2],mf2[i][j][k][3] = mse,psnr,ssim,retime
                print('OK',size[j],rate[k],dataset[i],i,j,k)
            except:
                print('error',size[j],rate[k],dataset[i],i,j,k)
            try:
                mse,psnr,ssim,retime = mm.f3(size[j],rate[k],dataset[i],0.1,20,64)
                mf3[i][j][k][0],mf3[i][j][k][1],mf3[i][j][k][2],mf3[i][j][k][3] = mse,psnr,ssim,retime
                print('OK',size[j],rate[k],dataset[i],i,j,k)
            except:
                print('error',size[j],rate[k],dataset[i],i,j,k)
            try:
                mse,psnr,ssim,retime = mm.f4(size[j],rate[k],dataset[i],0.1,20,64)
                mf4[i][j][k][0],mf4[i][j][k][1],mf4[i][j][k][2],mf4[i][j][k][3] = mse,psnr,ssim,retime
                print('OK',size[j],rate[k],dataset[i],i,j,k)
            except:
                print('error',size[j],rate[k],dataset[i],i,j,k)
            try:
                mse,psnr,ssim,retime = mm.f5(size[j],rate[k],dataset[i],0.1,20,64)
                mf5[i][j][k][0],mf5[i][j][k][1],mf5[i][j][k][2],mf5[i][j][k][3] = mse,psnr,ssim,retime
                print('OK',size[j],rate[k],dataset[i],i,j,k)
            except:
                print('error',size[j],rate[k],dataset[i],i,j,k)
            try:
                mse,psnr,ssim,retime = mm.f6(size[j],rate[k],dataset[i],0.1,20,64)
                mf6[i][j][k][0],mf6[i][j][k][1],mf6[i][j][k][2],mf6[i][j][k][3] = mse,psnr,ssim,retime
                print('OK',size[j],rate[k],dataset[i],i,j,k)
            except:
                print('error',size[j],rate[k],dataset[i],i,j,k)
            try:
                mse,psnr,ssim,retime = mm.f7(size[j],rate[k],dataset[i],0.1,20,64)
                mf7[i][j][k][0],mf7[i][j][k][1],mf7[i][j][k][2],mf7[i][j][k][3] = mse,psnr,ssim,retime
                print('OK',size[j],rate[k],dataset[i],i,j,k)
            except:
                print('error',size[j],rate[k],dataset[i],i,j,k)

np.save('mf1.npy',mf1)
np.save('mf2.npy',mf2)
np.save('mf3.npy',mf3)
np.save('mf4.npy',mf4)
np.save('mf5.npy',mf5)
np.save('mf6.npy',mf6)
np.save('mf7.npy',mf7)

