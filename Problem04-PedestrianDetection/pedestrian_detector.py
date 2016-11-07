from adjust_gamma import AdjustGamma
from pyramid import Pyramid
import matplotlib.pyplot as plt

class PedestrianDetector:
	"""docstring for PedestrianDetector"""
	def __init__(self, images):
		self.images = images
		self.configurations = {
			'gamma': 2.0
		}

	def train(self):
		for image in self.images:
			gamma_corrected = AdjustGamma.call(image, self.configurations['gamma'])
			for img in Pyramid.call(image):
				print img.shape
				raw_input("Wait")
		