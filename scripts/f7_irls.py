import math
import time
import  numpy as np    #对应numpy包
from PIL import Image  #对应pillow包
import matplotlib.pyplot as plt
import image_evaluate as ievalue
import os
import time

def irls(size,rate,raw_dataset,train_test_rate,epoch_num,batchsize):
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
    plt.show()
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


rate = 0.5
epoch_num = 20
batchsize = 64
size = 32
train_test_rate = 0.1
raw_dataset = 'stl'

mse,psnr,ssim,recovery_time = irls(size,rate,raw_dataset,train_test_rate,epoch_num,batchsize)
print(mse,psnr,ssim,recovery_time)






    

