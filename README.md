# Single_Pixel_Camera
Single pixel imaging system with deep learning

# Main Problem
1. Data is not enough   >>  100000
2. Data should be in the same category, otherwise the recovery accurate can be affected.
# New Idea
We can also use the decoded new images to do classification and compare the classification accuracy between raw images and decodes images. And use this to evaluate our reconstruction method.
# How to evaluate (1 & 2 are for face/cifar/stl  ;   3 is for mnist)
1. PSNR, SSIM, MSE(no mnist)
2. Reconstruction time(not necessary for sample time: sample time --> number of masks --> sample rate)   (no mnist)
3. Later Classification Performance(only for mnist)
# Experiment
1---------------------  exp_1
First get the best batchsize by testing(best batchsize is 2) and best matrix for the matrix method(best matrix is hadamard matrix)
Dataset: Face    Cifar   Stl
Size:    32      64      128
Rate:    0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
Method: f1-f7

Compare: PSNR, SSIM, MSE, Reconstruction Time
2--------------------   exp_2
Dataset: Mnist   size = 28    Need to save images and decoded images
Rate:    0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
Method: f1-f7

Compare: Classification rate