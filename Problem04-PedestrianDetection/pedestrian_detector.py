from hog import HOG
from pyramid import Pyramid
from windowize import Windowize
# import matplotlib.pyplot as plt

class PedestrianDetector:
	"""docstring for PedestrianDetector"""
	def __init__(self, images):
		self.images = images
		self.conigurations = {
			'minPyramidSize': (128, 64),
			'windowSize': (128, 64),
			'windowDisplacement': 8
		}

	def train(self):
		hog = HOG()
		for image in self.images:
			for img in self._pyramidize(image):
				for window in self._windownize(img):

	def _pyramidize(self, image):
		return Pyramid.call(
			image, minSize=self.configurations['minPyramidSize']
		)

	def _windownize(self, image):
		return Windownize.call(
			image,
			self.configurations['windowSize'],
			self.configurations['windowDisplacement']
		)