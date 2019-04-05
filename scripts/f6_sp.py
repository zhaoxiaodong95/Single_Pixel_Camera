import math
import time
import  numpy as np    #对应numpy包
from PIL import Image  #对应pillow包
import matplotlib.pyplot as plt
import image_evaluate as ievalue
import os
import time

def sp(size,rate,raw_dataset,train_test_rate,epoch_num,batchsize):
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
    plt.show()
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

rate = 0.5
epoch_num = 20
batchsize = 64
size = 32
train_test_rate = 0.1
raw_dataset = 'stl'

mse,psnr,ssim,recovery_time = sp(size,rate,raw_dataset,train_test_rate,epoch_num,batchsize)
print(mse,psnr,ssim,recovery_time)




    

