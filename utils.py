from scipy.fftpack import fft, ifft
from scipy.fftpack import fft2, ifft2
import tensorflow as tf
import cv2
import numpy as np




# def FR(image, mask):
image = cv2.imread("./001.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mask = cv2.imread("./radial_10.tif")
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask = mask/255
mask_1 = np.empty([256,256])
mask_1.fill(255)


# frq1 = tf.signal.fft2d(image)
# print(type(frq1))
# inv1 = tf.signal.ifft2d(frq1)

# frq2 = fft2(image)
# print(type(frq2))
# inv2 = ifft2(frq2)

# frq3 = tf.signal.fft(image)
# print(type(frq3))
# inv3 = tf.signal.ifft(frq3)

# frq4 = fft(image)
# print(type(frq4))
# inv4 = ifft(frq4)

# frq5 = np.fft.fft2(image)
# print(type(frq5))
# inv5 = np.fft.ifft2(frq5)

def RF(image, mask):
	img_complex = np.complex64(image)
	mask_complex = np.complex64(mask)

	# frq = tf.signal.fft2d(img_complex)
	frq = fft2(img_complex)
	frq = np.array(frq)
	res = np.multiply(frq, mask)
	return res


def FhRh(frq, mask):
	frq_mask = np.multiply(frq, mask)

	# res = tf.signal.ifft2d(frq_mask)
	res = ifft2(frq_mask)
	res = np.real(res)
	return res





# def tf_complex(data, name='tf_channel'):
# 	with tf.compat.v1.variable_scope(name+'_scope'):
# 		real  = np.float64(data[:,0:1,...])
# 		imag  = np.float64(data[:,1:2,...])
# 		del data
# 		data  = tf.complex(real, imag) 
# 	data = tf.identity(data, name=name)
# 	return data	