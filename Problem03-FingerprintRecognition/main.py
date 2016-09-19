from rindex28_loader import Rindex28Loader
import numpy as np
import cv2

def image_enhancement(image):
	mean = np.mean(image)
	std = np.std(image)
	image_enhanced = 150 + 95 * ((image - mean)/std)
	wrong_indexes = np.where(image_enhanced > 255)
	image_enhanced[wrong_indexes] = 255
	wrong_indexes = np.where(image_enhanced < 0)
	image_enhanced[wrong_indexes] = 0
	return np.array(image_enhanced, dtype=np.uint8)

rindex28 = Rindex28Loader('./databases/rindex28')
for image in rindex28.images:
	image_enhanced = image_enhancement(image)
	cv2.imshow('image', image_enhanced)
	cv2.waitKey(0)