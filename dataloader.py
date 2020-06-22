import cv2
import os
import numpy as np
from scipy.fftpack import fft2, ifft2
from utils import *


DATASET = 'brain'	#['brain', 'knees']
MASK = 'radial'		#['cartes', 'gauss', 'radial', 'spiral']
MASK_PRECENT = '10'	#['10', '20', '30', ... , '90']


datapath_train = './data/{}/db_train'.format(DATASET)
datapath_val = './data/{}/db_valid'.format(DATASET)
mask_path = './data/mask/{}/{}_{}.tif'.format(MASK, MASK, MASK_PRECENT)

us_train_data_ = []		#undersampled
us_val_data_ = []
train_data_ = []
val_data_ = []


for filename in os.listdir(datapath_train):
	image = cv2.imread('{}/{}'.format(datapath_train, filename))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	mask = cv2.imread(mask_path)
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	mask = mask/255

	frq = RF(image, mask)
	res = FhRh(frq)
	real = np.real(res)
	imag = np.imag(res)

	us_train_data_.append([real, imag])
	us_train_data = np.array(us_train_data_)

	image_imag = np.zeros(np.shape(image))
	train_data_.append([image, image_imag])
	train_data = np.array(train_data_)


for filename in os.listdir(datapath_val):
	image = cv2.imread('{}/{}'.format(datapath_val, filename))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	mask = cv2.imread(mask_path)
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	mask = mask/255

	frq = RF(image, mask)
	res = FhRh(frq)
	real = np.real(res)
	imag = np.imag(res)

	us_val_data_.append([real, imag])
	us_val_data = np.array(us_val_data_)

	image_imag = np.zeros(np.shape(image))
	val_data_.append([image, image_imag])
	val_data = np.array(val_data_)



us_min_data = np.min([us_train_data, us_val_data])
us_max_data = np.max([us_train_data, us_val_data])

us_train_data = 2*(us_train_data-us_min_data)/(us_max_data-us_min_data)-1
us_val_data = 2*(us_val_data-us_min_data)/(us_max_data-us_min_data)-1


min_data = np.min([train_data, val_data])
max_data = np.max([train_data, val_data])

train_data = 2*(train_data-min_data)/(max_data-min_data)-1
val_data = 2*(val_data-min_data)/(max_data-min_data)-1

