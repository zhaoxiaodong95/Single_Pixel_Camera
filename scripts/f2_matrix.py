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
from scipy.linalg import hadamard, toeplitz
import time

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

def weight_train_predict(size,rate,raw_dataset,matrix,train_test_rate,epoch_num,batchsize):
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
    plt.show()
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

rate = 0.1
epoch_num = 20
batchsize = 100
size = 32
train_test_rate = 0.1
raw_dataset = 'stl'
# matrix = 'PartHadamard'
matrix = ['SparseRandom','Bernoulli','PartHadamard','Toeplitz']
for metrix in matrix:

    mse,psnr,ssim,recovery_time = weight_train_predict(size,rate,raw_dataset,metrix,train_test_rate,epoch_num,batchsize)
    print(mse,psnr,ssim,recovery_time)