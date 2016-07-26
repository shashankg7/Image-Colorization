import glob , os , re
import numpy as np
from PIL import Image
from skimage.color import rgb2luv

from keras.models import Sequential
from keras.layers import Convolution2D 
from keras.layers import MaxPooling2D


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def preprocess(path):
	image = np.array((Image.open(path)).convert('L'))
	image = image.reshape((1,1,500,333))
	truthimage = np.array(Image.open(path))
	truthimage = rgb2luv(truthimage) #CIELUV Color Space
	return truthimage,image

def main():

	model = Sequential()
	model.add(Convolution2D(64, 3,3, border_mode='same', input_shape=(1,500,333),activation = 'relu'))
	model.add(Convolution2D(64, 3,3, border_mode='same', activation = 'relu'))
	model.add(MaxPooling2D())

	model.add(Convolution2D(128, 3,3, border_mode='same',activation = 'relu'))
	model.add(Convolution2D(128, 3,3, border_mode='same', activation = 'relu'))
	model.add(MaxPooling2D())

	model.add(Convolution2D(256, 3,3, border_mode='same',activation = 'relu'))
	model.add(Convolution2D(256, 3,3, border_mode='same', activation = 'relu'))
	model.add(Convolution2D(256, 3,3, border_mode='same', activation = 'relu'))
	model.add(MaxPooling2D())

	model.add(Convolution2D(512, 3,3, border_mode='same',activation = 'relu'))
	model.add(Convolution2D(512, 3,3, border_mode='same', activation = 'relu'))
	model.add(Convolution2D(512, 3,3, border_mode='same', activation = 'relu'))
	model.add(MaxPooling2D())

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	print "Model Compiled"


	path_to_dataset = "/home/himanshu/code/color/ILSVRC2015/Data/DET/"
	training_files = sorted(glob.glob(os.path.join(path_to_dataset,'test/*.JPEG')) , key = numericalSort)
	for idx , path in enumerate(training_files): #traverse the training images
		if idx < 1:
			truthimage, image = preprocess(path)
			print image.shape , truthimage.shape
			output = model.predict(image)
			print output.shape


if __name__ == '__main__':
	main()
	
	

