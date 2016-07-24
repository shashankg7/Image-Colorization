import glob
import os
import numpy as np
from PIL import Image
import cPickle
import re
from skimage.color import rgb2luv
from keras.models import Sequential
from keras.layers import Convolution2D


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def preprocess(path):
	image = np.array((Image.open(path)).convert('L'))
	image = image.reshape((2272,1704,1))
	truthimage = np.array(Image.open(path))
	truthimage = rgb2luv(truthimage) #CIELUV Color Space
	return truthimage,image

def create_nn():

	model = Sequential()
	model.add(Convolution2D(64, 3,3, border_mode='same', input_shape=(2272, 1704,1),activation = 'relu'))
	model.add(Convolution2D(64, 3,3, border_mode='same', input_shape=(2272, 1704,1),activation = 'relu'))



if __name__ == '__main__':
	
	path_to_dataset = "/home/himanshu/code/color/"
	training_files = sorted(glob.glob(os.path.join(path_to_dataset,'Train400Img/*.jpg')) , key = numericalSort)
	create_nn()
	for idx , path in enumerate(training_files): #traverse the training images
		truthimage, image = preprocess(path)
		print image.shape , truthimage.shape
