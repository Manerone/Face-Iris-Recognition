from skimage.transform import pyramid_gaussian
import cv2

class Pyramid:
	"""docstring for Pyramid"""
	@staticmethod
	def call(image, downscale=2, minSize=(128, 64)):
		for (i, resized) in enumerate(pyramid_gaussian(image, downscale=downscale)):
			# if the image is too small, break from the loop
			width, heigth, channels = resized.shape
			if width < minSize[0] or heigth < minSize[1]:
				break
				
			# show the resized image
			yield resized
		