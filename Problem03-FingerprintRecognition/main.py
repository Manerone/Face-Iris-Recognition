from rindex28_loader import Rindex28Loader
import numpy as np
import cv2
from matplotlib import pyplot

def image_enhancement(image):
	mean = np.mean(image)
	std = np.std(image)
	image_enhanced = 150 + 95 * ((image - mean)/std)
	wrong_indexes = np.where(image_enhanced > 255)
	image_enhanced[wrong_indexes] = 255
	wrong_indexes = np.where(image_enhanced < 0)
	image_enhanced[wrong_indexes] = 0
	return np.array(image_enhanced, dtype=np.uint8)

def orientation_computation(image):
	orientations = []
	# http://stackoverflow.com/questions/11105375/how-to-split-a-matrix-into-4-blocks-using-numpy
	# for each block of ten find the mean angle

rindex28 = Rindex28Loader('./databases/rindex28')
for image in rindex28.images:
	image_enhanced = image_enhancement(image)
	blurred_image = cv2.medianBlur(image_enhanced, 5)
	# cv2.imshow('image_e', image_enhanced)
	# cv2.waitKey(0)
	# cv2.imshow('image_b', blurred_image)
	# cv2.waitKey(0)
	orientations = orientation_computation(blurred_image)
