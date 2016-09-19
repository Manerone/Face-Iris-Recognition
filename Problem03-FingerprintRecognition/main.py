from rindex28_loader import Rindex28Loader
import cv2

def image_enhancement(image):
	mean, std = cv2.meanStdDev(image)
	i_max, j_max = image.shape
	for i in xrange(i_max):
		for j in xrange(j_max):
			value = 150 + 95 * ((image[i,j] - mean)/std)
			if value > 255:
				value = 255
			elif value < 0:
				value = 0
			image[i,j] = value

rindex28 = Rindex28Loader('./databases/rindex28')
for image in rindex28.images:
	image = image_enhancement(image)
	cv2.imshow('image', image)
	cv2.waitKey(0)