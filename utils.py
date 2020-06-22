# from scipy.fftpack import fft, ifft
from scipy.fftpack import fft2, ifft2
# import tensorflow as tf2
import cv2
import numpy as np




# def FR(image, mask):
# image = cv2.imread("./001.png")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# mask = cv2.imread("./radial_10.tif")
# mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# mask = mask/255
# mask_1 = np.empty([256,256])
# mask_1.fill(255)


def RF(image, mask):
	img_complex = np.complex64(image)
	mask_complex = np.complex64(mask)

	# frq = tf.signal.fft2d(img_complex)
	frq = fft2(img_complex)
	frq = np.array(frq)
	res = np.multiply(frq, mask)
	real = np.real(res)
	imag = np.imag(res)

	return res
	# return np.stack((real, imag), axis=0)	# returns (2,256,256)
	# return np.stack((real, imag), axis=2)	# returns (256,256,2)
	

def FhRh(frq):
	# frq_mask = np.multiply(frq, mask)

	# res = tf.signal.ifft2d(frq_mask)
	res = ifft2(frq)
	# res = np.real(res)
	return res
