import math
import time
import  numpy as np    #对应numpy包
from PIL import Image  #对应pillow包
import os
import matplotlib.pyplot as plt
import image_evaluate as ievalue
import time

def omp(size,rate,raw_dataset,train_test_rate,epoch_num,batchsize):
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
    plt.show()
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

rate = 0.001
epoch_num = 20
batchsize = 64
size = 32
train_test_rate = 0.1
raw_dataset = 'stl'

mse,psnr,ssim,recovery_time = omp(size,rate,raw_dataset,train_test_rate,epoch_num,batchsize)
print(mse,psnr,ssim,recovery_time)



    

