import glob
import os
import numpy as np
from PIL import Image
import cPickle
import re

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def get_images(files):
	# images = []
	for idx , path in enumerate(files):
		print idx
		image = (Image.open(path)).convert('L')
		images.append(image)
	return images



if __name__ == '__main__':
	
	path_to_dataset = "/home/himanshu/code/color/"
	training_files = sorted(glob.glob(os.path.join(path_to_dataset,'Train400Img/*.jpg')) , key = numericalSort)

	for idx , path in enumerate(training_files): #traverse the training images
		image = np.array(Image.open(path))
		print image.shape
		print type(image)