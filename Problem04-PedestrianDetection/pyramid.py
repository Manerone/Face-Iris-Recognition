from skimage.transform import pyramid_gaussian

class Pyramid:
	"""docstring for Pyramid"""
	@staticmethod
	def call(image, downscale=2, minSize=(128, 64)):
		for (i, resized) in enumerate(pyramid_gaussian(image, downscale=downscale)):
			# if the image is too small, break from the loop
			heigth, width, channels = resized.shape
			if heigth < minSize[0] or width < minSize[1]:
				break
				
			# show the resized image
			yield resized
		