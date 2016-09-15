import glob , os , re
import numpy as np
from PIL import Image
from skimage.color import rgb2luv
import pdb


from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D
from keras.layers import MaxPooling2D , UpSampling2D
from keras.layers import Merge


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def preprocess(path):

	image = np.array((Image.open(path)).convert('L'))
	pdb.set_trace()
	image = image.reshape((1,1,512,336)) #500x333 being size of image

	truthimage = np.array(Image.open(path))
	truthimage = rgb2luv(truthimage) #CIELUV Color Space

	return truthimage,image

def main():

	model10 = Sequential()
	model10.add(Convolution2D(64, 3,3, border_mode='same', input_shape=(1,512,336),activation = 'relu'))
	model10.add(Convolution2D(64, 3,3, border_mode='same', activation = 'relu'))


	model20 = Sequential()
	model20.add(model10)
	model20.add(MaxPooling2D())
	model20.add(Convolution2D(128, 3,3, border_mode='same',activation = 'relu'))
	model20.add(Convolution2D(128, 3,3, border_mode='same', activation = 'relu'))

	model21 = Sequential()
	model21.add(model20)
	model21.add(MaxPooling2D())

	model30 = Sequential()
	model30.add(model20)
	model30.add(MaxPooling2D())
	model30.add(Convolution2D(256, 3,3, border_mode='same',activation = 'relu'))
	model30.add(Convolution2D(256, 3,3, border_mode='same', activation = 'relu'))
	model30.add(Convolution2D(256, 3,3, border_mode='same', activation = 'relu'))

	model40 = Sequential()
	model40.add(model30)
	model40.add(MaxPooling2D())
	model40.add(Convolution2D(512, 3,3, border_mode='same',activation = 'relu'))
	model40.add(Convolution2D(512, 3,3, border_mode='same', activation = 'relu'))
	model40.add(Convolution2D(512, 3,3, border_mode='same', activation = 'relu'))

	model40.add(Convolution2D(256, 1,1, border_mode='same', activation = 'relu'))
	model40.add(UpSampling2D(size=(2, 2), dim_ordering='th'))


	model41 = Sequential()
	model41.add(Merge([model40,model30] , mode = 'sum'))
	model41.add(Convolution2D(128, 3,3, border_mode='same', activation = 'relu'))

	model42 = Sequential()
	model42.add(Merge([model41,model21] , mode = 'sum'))
	model42.add(Convolution2D(64, 3,3, border_mode='same', activation = 'relu'))

	model50 = Sequential()
	model50.add(Merge([model10,model41,model42,model40],mode = 'concat'))
	model50.add(Convolution2D(256, 3,3, border_mode='same',activation = 'relu'))
	model50.add(Convolution2D(64, 3,3, border_mode='same', activation = 'relu'))
	model50.add(Convolution2D(64, 3,3, border_mode='same', activation = 'relu'))

	model50.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	print "Model Compiled"


	path_to_dataset = "./trial"
	training_files = sorted(glob.glob(os.path.join(path_to_dataset,'*.JPEG')) , key = numericalSort)
	for idx , path in enumerate(training_files): #traverse the training images
		if idx < 1:
			truthimage, image = preprocess(path)
			print image.shape , truthimage.shape
			output = model30.predict(image)
			print output.shape


if __name__ == '__main__':
	main()