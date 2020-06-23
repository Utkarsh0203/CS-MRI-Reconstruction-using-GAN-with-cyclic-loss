import cv2
import os
import numpy as np
from scipy.fftpack import fft2, ifft2
from utils import *
import pickle as pkl


DATASET = 'brain'	#['brain', 'knees']
MASK = 'radial'		#['cartes', 'gauss', 'radial', 'spiral']
MASK_PRECENT = '10'	#['10', '20', '30', ... , '90']


datapath_train = './data/{}/db_train'.format(DATASET)
datapath_val = './data/{}/db_valid'.format(DATASET)
mask_path = './data/mask/{}/{}_{}.tif'.format(MASK, MASK, MASK_PRECENT)
PIK = './data.pkl'
MM = './minmax.pkl'


mask = cv2.imread(mask_path)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask = mask/255
image_imag = np.zeros([256, 256])


us_train_data = np.zeros([100, 2, 256, 256])		#undersampled
us_val_data = np.zeros([100, 2, 256, 256])			
train_data = np.zeros([100, 2, 256, 256])			#original image
val_data = np.zeros([100, 2, 256, 256])			
RF_train_data = np.zeros([100, 2, 256, 256])		#in k-space
RF_val_data = np.zeros([100, 2, 256, 256])			

i=0
for filename in os.listdir(datapath_train):
	image = cv2.imread('{}/{}'.format(datapath_train, filename))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	frq = RF(image, mask)
	RF_train_data[i,0] = np.real(frq)
	RF_train_data[i,1] = np.imag(frq)

	res = FhRh(frq)
	us_train_data[i,0] = np.real(res)
	us_train_data[i,1] = np.imag(res)

	train_data[i,0] = image
	train_data[i,1] = image_imag
	i = i+1

j=0
for filename in os.listdir(datapath_val):
	image = cv2.imread('{}/{}'.format(datapath_val, filename))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	frq = RF(image, mask)
	RF_val_data[j,0] = np.real(frq)
	RF_val_data[j,1] = np.imag(frq)

	res = FhRh(frq)
	us_val_data[j,0] = np.real(res)
	us_val_data[j,1] = np.imag(res)

	val_data[j,0] = image
	val_data[j,1] = image_imag
	j = j+1


min_us_data = np.min([us_train_data, us_val_data])
max_us_data = np.max([us_train_data, us_val_data])
us_train_data = 2*(us_train_data-min_us_data)/(max_us_data-min_us_data)-1
us_val_data = 2*(us_val_data-min_us_data)/(max_us_data-min_us_data)-1


min_data = np.min([train_data, val_data])
max_data = np.max([train_data, val_data])
train_data = 2*(train_data-min_data)/(max_data-min_data)-1
val_data = 2*(val_data-min_data)/(max_data-min_data)-1


min_RF_data = np.min([RF_train_data, RF_val_data])
max_RF_data = np.max([RF_train_data, RF_val_data])
RF_train_data = 2*(RF_train_data-min_RF_data)/(max_RF_data-min_RF_data)-1
RF_val_data = 2*(RF_val_data-min_RF_data)/(max_RF_data-min_RF_data)-1


DATA = [us_train_data, us_val_data, train_data, val_data, RF_train_data, RF_val_data]
MINMAX = [min_us_data, max_us_data, min_data, max_data, min_RF_data, max_RF_data]


def save_data():
	with open(PIK, "wb") as f:
		pkl.dump(DATA, f)

	with open(MM, "wb") as g:
		pkl.dump(MINMAX, g)
