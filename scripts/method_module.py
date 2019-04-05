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
from scipy.linalg import hadamard, toeplitz
import math

def f1(size,rate,raw_dataset,train_test_rate,epoch_num,batchsize):
    
    i = 0
    files = os.listdir(raw_dataset)  
    print(len(files))
    files.sort()
    pix = np.zeros(shape=(len(files),size,size))
    for file in files:
        filename = raw_dataset + '/'  + file
        img=np.array(Image.open(filename).convert('L'))
        pia = img[int((img.shape[0]-size)/2):int((img.shape[0]-size)/2+size)]
        pib = pia[:,int((img.shape[1]-size)/2):int((img.shape[1]-size)/2+size)]
        pix[i]=pib
        i = i + 1

    x_train = np.zeros(shape=(int(len(files) * (1-train_test_rate)),size,size))
    x_test = np.zeros(shape=(int(len(files) * (train_test_rate)),size,size))

    for i in range(int(len(files) * (1-train_test_rate))):
        x_train[i] = pix[i]
    for i in range(int(len(files) * (train_test_rate))):
        x_test[i] = pix[i+int(len(files) * (1-train_test_rate))]

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), size,size, 1))
    x_test = np.reshape(x_test, (len(x_test),size,size, 1))

    print(x_train.shape)
    print(x_test.shape)

    M =int(size*size*rate)

    input_img = Input(shape=(size,size,1))
    xx = Reshape((size * size,), input_shape=(size,size,1))(input_img)
    #if the weight > 0, weight = 1, otherwise weight = 0
    encoded = Dense(M,activation='relu',use_bias = False, kernel_constraint = two_value())(xx)
    encoder = Model(inputs=input_img, outputs=encoded)
    #######################################################################
    x = Dense(size * size,activation='relu')(encoded)
    x = Reshape((size,size))(x)
    x = LSTM(units=12)(x)
    x = Dense(size * size,activation='relu')(x)
    x = Reshape((size,size,1))(x)
    x = Convolution2D(32, (2, 2), activation='relu', padding='same')(x)
    x = Convolution2D(16, (2, 2), activation='relu', padding='same')(x)
    decoded = Convolution2D(1, (2, 2), activation='sigmoid', padding='same')(x)
    ######################################################################
    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(optimizer='RMSprop', loss='binary_crossentropy')
    # 打开一个终端并启动TensorBoard，终端中输入 tensorboard --logdir ./autoencoder
    autoencoder.fit(x_train, x_train, epochs=epoch_num, batch_size=batchsize,)
    recovery_start = time.time()
    decoded_imgs = autoencoder.predict(x_test)
    recovery_end = time.time()


    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(size,size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
 
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(size,size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # plt.show()
    mse = np.zeros(int(len(files) * (train_test_rate)))
    psnr = np.zeros(int(len(files) * (train_test_rate)))
    ssim = np.zeros(int(len(files) * (train_test_rate)))
    x_axis = np.arange(int(len(files) * (train_test_rate)))
    for i in range(int(len(files) * (train_test_rate))):
        array1 = x_test[i].reshape(size,size)
        array1 = array1*255    #  
        array2 = decoded_imgs[i].reshape(size,size)
        array2 = array2*255    #   
        mse[i],psnr[i],ssim[i] = ievalue.cal_mse_psnr_ssim(array1,array2)  
    return np.mean(mse),np.mean(psnr),np.mean(ssim),(recovery_end-recovery_start)/int(len(files) * (train_test_rate))


def SparseRandom(m,n,seeds):
    np.random.seed(seeds)
    alpha = 0.05
    d = int(alpha*m)
    phi = np.zeros((m,n))
    for i in range(n):
        col = np.random.permutation(m)
        index = col[0:d]
        # print('index',index[1])
        for j in range(d):
            phi[index[j]][i] = 1
    return phi
def Bernoulli(m,n,seeds):
    np.random.seed(seeds)
    phi = np.random.randint(2,size=(m,n))
    phi[phi==0]=-1
    return phi
def PartHadamard(m,n,seeds):
    np.random.seed(seeds)
    lt = np.max([m,n])
    lt1 = (12-np.mod(lt,12)) + lt
    lt2 = (20-np.mod(lt,20)) + lt
    lt3 = np.power(2,np.ceil(np.log2(lt)))
    l = np.min([lt1,lt2,lt3])
    l = int(l)
    phi = np.array((m,n))
    phit = hadamard(l)
    row = np.random.permutation(l)
    col = np.random.permutation(l)
    rowindex = row[0:m]
    colindex = col[0:n]
    phi_1 = phit[rowindex]
    phi_2 = phi_1[:,colindex]
    return phi_2
def Toeplitz(m,n,seeds):
    np.random.seed(seeds)
    u = np.random.randint(2,size=(1,2*n-1))
    u[u==0] = -1
    phit = toeplitz(u[:,n-1:],np.fliplr(u[:,0:n]))
    print(phit)
    phi = phit[0:m,:]
    return phi

def f2(size,rate,raw_dataset,matrix,train_test_rate,epoch_num,batchsize):
    M =int(size*size*rate)
    imgab = size*size
    if matrix == 'SparseRandom':
        H2 = SparseRandom(M,imgab,66)
    elif matrix == 'Bernoulli':
        H2 = Bernoulli(M,imgab,66)
    elif matrix == 'PartHadamard':
        H2 = PartHadamard(M,imgab,66)
    elif matrix == 'Toeplitz':
        H2 = Toeplitz(M,imgab,66)
    print(H2)

    i = 0
    files = os.listdir(raw_dataset)  
    print(len(files))
    files.sort()
    pix = np.zeros(shape=(len(files),size,size))
    for file in files:
        filename = raw_dataset + '/'  + file
        img=np.array(Image.open(filename).convert('L'))
        pia = img[int((img.shape[0]-size)/2):int((img.shape[0]-size)/2+size)]
        pib = pia[:,int((img.shape[1]-size)/2):int((img.shape[1]-size)/2+size)]
        pix[i]=pib
        i = i + 1

    x_train = np.zeros(shape=(int(len(files) * (1-train_test_rate)),size,size))
    x_test = np.zeros(shape=(int(len(files) * (train_test_rate)),size,size))
    x_train_input = np.zeros([int(len(files) * (1-train_test_rate)),M])
    x_test_input = np.zeros([int(len(files) * (train_test_rate)),M])

    for i in range(int(len(files) * (1-train_test_rate))):
        x_train[i] = pix[i]
        x_train_input[i] = np.matmul(H2,x_train[i].reshape((imgab,)))
    for i in range(int(len(files) * (train_test_rate))):
        x_test[i] = pix[i+int(len(files) * (1-train_test_rate))]
        x_test_input[i] = np.matmul(H2,x_test[i].reshape((imgab,)))

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), size,size, 1))
    x_test = np.reshape(x_test, (len(x_test),size,size, 1))

    print(x_train.shape)
    print(x_test.shape)

    

    input_img = Input(shape=(M,))
    #######################################################################
    x = Dense(size * size,activation='relu')(input_img)
    x = Reshape((size,size))(x)
    x = LSTM(units=12)(x)
    x = Dense(size * size,activation='relu')(x)
    x = Reshape((size,size,1))(x)
    x = Convolution2D(32, (2, 2), activation='relu', padding='same')(x)
    x = Convolution2D(16, (2, 2), activation='relu', padding='same')(x)
    decoded = Convolution2D(1, (2, 2), activation='sigmoid', padding='same')(x)
    ######################################################################
    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(optimizer='RMSprop', loss='binary_crossentropy')
    # 打开一个终端并启动TensorBoard，终端中输入 tensorboard --logdir ./autoencoder
    autoencoder.fit(x_train_input, x_train, epochs=epoch_num, batch_size=batchsize,)
    recovery_start = time.time()
    decoded_imgs = autoencoder.predict(x_test_input)
    recovery_end = time.time()
  

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(size,size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
 
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(size,size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # plt.show()
    mse = np.zeros(int(len(files) * (train_test_rate)))
    psnr = np.zeros(int(len(files) * (train_test_rate)))
    ssim = np.zeros(int(len(files) * (train_test_rate)))
    x_axis = np.arange(int(len(files) * (train_test_rate)))
    for i in range(int(len(files) * (train_test_rate))):
        array1 = x_test[i].reshape(size,size)
        array1 = array1*255    #  
        array2 = decoded_imgs[i].reshape(size,size)
        array2 = array2*255    #   
        mse[i],psnr[i],ssim[i] = ievalue.cal_mse_psnr_ssim(array1,array2)  
    return np.mean(mse),np.mean(psnr),np.mean(ssim),(recovery_end-recovery_start)/int(len(files) * (train_test_rate))

def f3(size,rate,raw_dataset,train_test_rate,epoch_num,batchsize):
    
    i = 0
    files = os.listdir(raw_dataset)  
    print(len(files))
    files.sort()
    pix = np.zeros(shape=(len(files),size,size))
    for file in files:
        filename = raw_dataset + '/'  + file
        img=np.array(Image.open(filename).convert('L'))
        pia = img[int((img.shape[0]-size)/2):int((img.shape[0]-size)/2+size)]
        pib = pia[:,int((img.shape[1]-size)/2):int((img.shape[1]-size)/2+size)]
        pix[i]=pib
        i = i + 1

    x_train = np.zeros(shape=(int(len(files) * (1-train_test_rate)),size,size))
    x_test = np.zeros(shape=(int(len(files) * (train_test_rate)),size,size))

    for i in range(int(len(files) * (1-train_test_rate))):
        x_train[i] = pix[i]
    for i in range(int(len(files) * (train_test_rate))):
        x_test[i] = pix[i+int(len(files) * (1-train_test_rate))]

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), size,size, 1))
    x_test = np.reshape(x_test, (len(x_test),size,size, 1))

    print(x_train.shape)
    print(x_test.shape)

    M =int(size*size*rate)

    input_img = Input(shape=(size,size,1))
    xx = Reshape((size * size,), input_shape=(size,size,1))(input_img)
    #if the weight > 0, weight = 1, otherwise weight = 0
    encoded = Dense(M,activation='relu',use_bias = False, kernel_constraint = two_value())(xx)
    encoder = Model(inputs=input_img, outputs=encoded)
    #######################################################################
    x = Dense(size * size,activation='relu')(encoded)
    x = Reshape((size,size,1))(x)
    x = Convolution2D(32, (2, 2), activation='relu', padding='same')(x)
    x = Convolution2D(16, (2, 2), activation='relu', padding='same')(x)
    decoded = Convolution2D(1, (2, 2), activation='sigmoid', padding='same')(x)
    ######################################################################
    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(optimizer='RMSprop', loss='binary_crossentropy')
    # 打开一个终端并启动TensorBoard，终端中输入 tensorboard --logdir ./autoencoder
    autoencoder.fit(x_train, x_train, epochs=epoch_num, batch_size=batchsize,)
    recovery_start = time.time()
    decoded_imgs = autoencoder.predict(x_test)
    recovery_end = time.time()

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(size,size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
 
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(size,size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # plt.show()
    mse = np.zeros(int(len(files) * (train_test_rate)))
    psnr = np.zeros(int(len(files) * (train_test_rate)))
    ssim = np.zeros(int(len(files) * (train_test_rate)))
    x_axis = np.arange(int(len(files) * (train_test_rate)))
    for i in range(int(len(files) * (train_test_rate))):
        array1 = x_test[i].reshape(size,size)
        array1 = array1*255    #  
        array2 = decoded_imgs[i].reshape(size,size)
        array2 = array2*255    #   
        mse[i],psnr[i],ssim[i] = ievalue.cal_mse_psnr_ssim(array1,array2)  
    return np.mean(mse),np.mean(psnr),np.mean(ssim),(recovery_end-recovery_start)/int(len(files) * (train_test_rate))

def f4(size,rate,raw_dataset,train_test_rate,epoch_num,batchsize):
    
    i = 0
    files = os.listdir(raw_dataset)  
    print(len(files))
    files.sort()
    pix = np.zeros(shape=(len(files),size,size))
    for file in files:
        filename = raw_dataset + '/'  + file
        img=np.array(Image.open(filename).convert('L'))
        pia = img[int((img.shape[0]-size)/2):int((img.shape[0]-size)/2+size)]
        pib = pia[:,int((img.shape[1]-size)/2):int((img.shape[1]-size)/2+size)]
        pix[i]=pib
        i = i + 1

    x_train = np.zeros(shape=(int(len(files) * (1-train_test_rate)),size,size))
    x_test = np.zeros(shape=(int(len(files) * (train_test_rate)),size,size))

    for i in range(int(len(files) * (1-train_test_rate))):
        x_train[i] = pix[i]
    for i in range(int(len(files) * (train_test_rate))):
        x_test[i] = pix[i+int(len(files) * (1-train_test_rate))]

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), size,size, 1))
    x_test = np.reshape(x_test, (len(x_test),size,size, 1))

    print(x_train.shape)
    print(x_test.shape)

    M =int(size*size*rate)

    input_img = Input(shape=(size,size,1))
    xx = Reshape((size * size,), input_shape=(size,size,1))(input_img)
    #if the weight > 0, weight = 1, otherwise weight = 0
    encoded = Dense(M,activation='relu',use_bias = False, kernel_constraint = two_value())(xx)
    encoder = Model(inputs=input_img, outputs=encoded)
    #######################################################################
    x = Dense(size*size,activation='sigmoid')(encoded)
    x = Dense(size*size*2,activation='sigmoid')(x)
    x = Dense(size*size*8,activation='sigmoid')(x)
    x = Dense(size*size*2,activation='sigmoid')(x)
    x = Dense(size*size,activation='sigmoid')(x)
    decoded = Reshape((size,size,1))(x)
    ######################################################################
    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(optimizer='RMSprop', loss='binary_crossentropy')
    # 打开一个终端并启动TensorBoard，终端中输入 tensorboard --logdir ./autoencoder
    autoencoder.fit(x_train, x_train, epochs=epoch_num, batch_size=batchsize,)
    recovery_start = time.time()
    decoded_imgs = autoencoder.predict(x_test)
    recovery_end = time.time()

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(size,size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
 
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(size,size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # plt.show()
    mse = np.zeros(int(len(files) * (train_test_rate)))
    psnr = np.zeros(int(len(files) * (train_test_rate)))
    ssim = np.zeros(int(len(files) * (train_test_rate)))
    x_axis = np.arange(int(len(files) * (train_test_rate)))
    for i in range(int(len(files) * (train_test_rate))):
        array1 = x_test[i].reshape(size,size)
        array1 = array1*255    #  
        array2 = decoded_imgs[i].reshape(size,size)
        array2 = array2*255    #   
        mse[i],psnr[i],ssim[i] = ievalue.cal_mse_psnr_ssim(array1,array2)  
    return np.mean(mse),np.mean(psnr),np.mean(ssim),(recovery_end-recovery_start)/int(len(files) * (train_test_rate))

def f5(size,rate,raw_dataset,train_test_rate,epoch_num,batchsize):
    files = os.listdir(raw_dataset)  
    test_num = len(files)
    pix = np.zeros(shape=(len(files),size,size))
    count = 0
    for file in files:
        filename = raw_dataset + '/'  + file
        img=np.array(Image.open(filename).convert('L'))
        pia = img[int((img.shape[0]-size)/2):int((img.shape[0]-size)/2+size)]
        pib = pia[:,int((img.shape[1]-size)/2):int((img.shape[1]-size)/2+size)]
        pix[count]=pib
        count = count + 1
    omp_result = np.zeros((test_num,size,size))
    #OMP算法函数
    def cs_omp(y,D):    
        L=math.floor(3*(y.shape[0])/4)
        residual=y  #初始化残差
        index=np.zeros(size,dtype=int)
        for i in range(size):
            index[i]= -1
        result=np.zeros((size))
        for j in range(L):  #迭代次数
            product=np.fabs(np.dot(D.T,residual))
            pos=np.argmax(product)  #最大投影系数对应的位置  
                
            index[j]=pos
        
        
            my=np.linalg.pinv(D[:,index>=0]) #最小二乘,看参考文献1    
            
            a=np.dot(my,y) #最小二乘,看参考文献1     
            residual=y-np.dot(D[:,index>=0],a)
        result[index>=0]=a
        return  result
    #生成高斯随机测量矩阵
    # Phi=np.random.randn(int(size*sampleRate),size)
    Phi = np.random.randn(size, size)
    u, s, vh = np.linalg.svd(Phi)
    Phi = u[:int(size*rate),] #将测量矩阵正交化

    #生成稀疏基DCT矩阵
    mat_dct_1d=np.zeros((size,size))
    v=range(size)
    for k in range(0,size):  
        dct_1d=np.cos(np.dot(v,k*math.pi/size))
        if k>0:
            dct_1d=dct_1d-np.mean(dct_1d)
        mat_dct_1d[:,k]=dct_1d/np.linalg.norm(dct_1d)
    
    recovery_time = np.zeros(test_num)
    
    for j in range(test_num):
        print("the %s th image\n" % j)
        #随机测量
        im = pix[j]
        img_cs_1d=np.dot(Phi,im)
        recovery_start = time.time()
        #重建
        sparse_rec_1d=np.zeros((size,size))   # 初始化稀疏系数矩阵    
        Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵
        for i in range(size):
            # print('正在重建第',i,'列。')
            column_rec=cs_omp(img_cs_1d[:,i],Theta_1d) #利用OMP算法计算稀疏系数
            sparse_rec_1d[:,i]=column_rec;        
        img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵
        omp_result[j] = img_rec
        recovery_end = time.time()
        recovery_time[j] = recovery_end - recovery_start


    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(pix[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(omp_result[i])
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False) 
    # plt.show()
    mse = np.zeros(test_num)
    psnr = np.zeros(test_num)
    ssim = np.zeros(test_num)
    nouse = 0
    for i in range(test_num):
        array1 = pix[i].reshape(size,size)
        array1 = array1   #  
        array2 = omp_result[i].reshape(size,size)
        array2 = array2    #   
        m,p,s = ievalue.cal_mse_psnr_ssim(array1,array2)  
        if m==0:
            print('----------',i)
            nouse = nouse + 1
            continue
        mse[i],psnr[i],ssim[i] = m,p,s 
    return np.sum(mse)/(test_num-nouse),np.sum(psnr)/(test_num-nouse),np.sum(ssim)/(test_num-nouse),np.sum(recovery_time)/(test_num-nouse)
def f6(size,rate,raw_dataset,train_test_rate,epoch_num,batchsize):
    files = os.listdir(raw_dataset)  
    test_num = len(files)
    pix = np.zeros(shape=(len(files),size,size))
    count = 0
    for file in files:
        filename = raw_dataset + '/'  + file
        img=np.array(Image.open(filename).convert('L'))
        pia = img[int((img.shape[0]-size)/2):int((img.shape[0]-size)/2+size)]
        pib = pia[:,int((img.shape[1]-size)/2):int((img.shape[1]-size)/2+size)]
        pix[count]=pib
        count = count + 1
    sp_result = np.zeros((test_num,size,size))
    #SP算法函数
    def cs_sp(y,D):     
        K=math.floor(y.shape[0]/3)  
        pos_last=np.array([],dtype=np.int64)
        result=np.zeros((size))

        product=np.fabs(np.dot(D.T,y))
        pos_temp=product.argsort() 
        pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
        pos_current=pos_temp[0:K]#初始化索引集 对应初始化步骤1
        residual_current=y-np.dot(D[:,pos_current],np.dot(np.linalg.pinv(D[:,pos_current]),y))#初始化残差 对应初始化步骤2

        while True:  #迭代次数
            product=np.fabs(np.dot(D.T,residual_current))       
            pos_temp=np.argsort(product)
            pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
            pos=np.union1d(pos_current,pos_temp[0:K])#对应步骤1     
            pos_temp=np.argsort(np.fabs(np.dot(np.linalg.pinv(D[:,pos]),y)))#对应步骤2  
            pos_temp=pos_temp[::-1]
            pos_last=pos_temp[0:K]#对应步骤3    
            residual_last=y-np.dot(D[:,pos_last],np.dot(np.linalg.pinv(D[:,pos_last]),y))#更新残差 #对应步骤4
            if np.linalg.norm(residual_last)>=np.linalg.norm(residual_current): #对应步骤5  
                pos_last=pos_current
                break
            residual_current=residual_last
            pos_current=pos_last
        result[pos_last[0:K]]=np.dot(np.linalg.pinv(D[:,pos_last[0:K]]),y) #对应输出步骤  
        return  result

    #生成高斯随机测量矩阵

    Phi=np.random.randn(int(size*rate),size)

    #生成稀疏基DCT矩阵
    mat_dct_1d=np.zeros((size,size))
    v=range(size)
    for k in range(0,size):  
        dct_1d=np.cos(np.dot(v,k*math.pi/size))
        if k>0:
            dct_1d=dct_1d-np.mean(dct_1d)
        mat_dct_1d[:,k]=dct_1d/np.linalg.norm(dct_1d)
    recovery_time = np.zeros(test_num)
    for j in range(test_num):
        print("the %s th image\n" % j)
        #随机测量
        im = pix[j]
        start = time.clock()
        img_cs_1d=np.dot(Phi,im)
        recovery_start = time.time()
        #重建
        sparse_rec_1d=np.zeros((size,size))   # 初始化稀疏系数矩阵    
        Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵
        for i in range(size):
            
            column_rec=cs_sp(img_cs_1d[:,i],Theta_1d) #利用OMP算法计算稀疏系数
            sparse_rec_1d[:,i]=column_rec;        
        img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵
        sp_result[j] = img_rec
        recovery_end = time.time()
        #显示重建后的图片
        recovery_time[j] = recovery_end - recovery_start
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(pix[i])
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(sp_result[i])
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False) 
    # plt.show()
    mse = np.zeros(test_num)
    psnr = np.zeros(test_num)
    ssim = np.zeros(test_num)
    nouse = 0
    for i in range(test_num):
        array1 = pix[i].reshape(size,size)
        array1 = array1   #  
        array2 = sp_result[i].reshape(size,size)
        array2 = array2    #   
        m,p,s = ievalue.cal_mse_psnr_ssim(array1,array2)  
        if m==0:
            print('----------',i)
            nouse = nouse + 1
            continue
        mse[i],psnr[i],ssim[i] = m,p,s 
    return np.sum(mse)/(test_num-nouse),np.sum(psnr)/(test_num-nouse),np.sum(ssim)/(test_num-nouse),np.sum(recovery_time)/(test_num-nouse)

def f7(size,rate,raw_dataset,train_test_rate,epoch_num,batchsize):
    files = os.listdir(raw_dataset)  
    test_num = len(files)
    pix = np.zeros(shape=(len(files),size,size))
    count = 0
    for file in files:
        filename = raw_dataset + '/'  + file
        img=np.array(Image.open(filename).convert('L'))
        pia = img[int((img.shape[0]-size)/2):int((img.shape[0]-size)/2+size)]
        pib = pia[:,int((img.shape[1]-size)/2):int((img.shape[1]-size)/2+size)]
        pix[count]=pib
        count = count + 1
    irls_result = np.zeros((test_num,size,size))
    #IRLS算法函数
    def cs_irls(y,T_Mat):   
        L=math.floor((y.shape[0])/4)
        # print('----------------',L)
        hat_x_tp=np.dot(T_Mat.T ,y)
        epsilong=1
        p=1 # solution for l-norm p
        times=1
        while (epsilong>10e-9) and (times<L):  #迭代次数
            weight=(hat_x_tp**2+epsilong)**(p/2-1)
            Q_Mat=np.diag(1/weight)
            #hat_x=Q_Mat*T_Mat'*inv(T_Mat*Q_Mat*T_Mat')*y
            temp=np.dot(np.dot(T_Mat,Q_Mat),T_Mat.T)
            temp=np.dot(np.dot(Q_Mat,T_Mat.T),np.linalg.inv(temp))
            hat_x=np.dot(temp,y)        
            if(np.linalg.norm(hat_x-hat_x_tp,2) < np.sqrt(epsilong)/100):
                epsilong = epsilong/10
            hat_x_tp=hat_x
            times=times+1
        return hat_x

    #生成高斯随机测量s矩阵

    # Phi=np.random.randn(int(size*sampleRate),size)
    Phi = np.random.randn(size, size)
    u, s, vh = np.linalg.svd(Phi)
    Phi = u[:int(size*rate),] #将测量矩阵正交化

    #生成稀疏基DCT矩阵
    mat_dct_1d=np.zeros((size,size))
    v=range(size)
    for k in range(0,size):  
        dct_1d=np.cos(np.dot(v,k*math.pi/size))
        if k>0:
            dct_1d=dct_1d-np.mean(dct_1d)
        mat_dct_1d[:,k]=dct_1d/np.linalg.norm(dct_1d)
    recovery_time = np.zeros(test_num)
    for j in range(test_num):
        print("the %s th image\n" % j)
        #随机测量
        im = pix[j]
        start = time.clock()
        img_cs_1d=np.dot(Phi,im)
        recovery_start = time.time()
        #重建
        sparse_rec_1d=np.zeros((size,size))   # 初始化稀疏系数矩阵    
        Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵
        for i in range(size):
            # print('正在重建第',i,'列。')
            column_rec=cs_irls(img_cs_1d[:,i],Theta_1d)  #利用IRLS算法计算稀疏系数
            sparse_rec_1d[:,i]=column_rec;        
        img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵
        irls_result[j] = img_rec
        recovery_end = time.time()
        recovery_time[j] = recovery_end - recovery_start
        
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(pix[i])
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(irls_result[i])
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False) 
    # plt.show()
    mse = np.zeros(test_num)
    psnr = np.zeros(test_num)
    ssim = np.zeros(test_num)
    nouse = 0
    for i in range(test_num):
        array1 = pix[i].reshape(size,size)
        array1 = array1   #  
        array2 = irls_result[i].reshape(size,size)
        array2 = array2    #   
        m,p,s = ievalue.cal_mse_psnr_ssim(array1,array2)  
        if m==0:
            print('----------',i)
            nouse = nouse + 1
            continue
        mse[i],psnr[i],ssim[i] = m,p,s 
    return np.sum(mse)/(test_num-nouse),np.sum(psnr)/(test_num-nouse),np.sum(ssim)/(test_num-nouse),np.sum(recovery_time)/(test_num-nouse)




