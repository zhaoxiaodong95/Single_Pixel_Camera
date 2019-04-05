from PIL import Image
import numpy as np

def cal_mse_psnr_ssim(im1, im2):
	'''
	im1 and im2 should be numpy array
	'''
	# img1 = Image.open(file1).convert('L')
	# img2 = Image.open(file2).convert('L')

	# im1 = np.array(img1)
	# im2 = np.array(img2)

	mse = (np.abs(im1/1.0 - im2/1.0) ** 2).mean()
	psnr = 10 * np.log10(255 * 255 / mse)
	mu1 = im1.mean()
	mu2 = im2.mean()
	sigma1 = np.sqrt(((im1/1.0 - mu1/1.0) ** 2).mean())
	sigma2 = np.sqrt(((im2/1.0 - mu2/1.0) ** 2).mean())
	sigma12 = ((im1/1.0 - mu1/1.0) * (im2/1.0 - mu2/1.0)).mean()
	k1, k2, L = 0.01, 0.03, 255
	C1 = (k1*L) ** 2
	C2 = (k2*L) ** 2
	C3 = C2/2
	l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
	c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
	s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
	ssim = l12 * c12 * s12
	return mse,psnr,ssim






    

