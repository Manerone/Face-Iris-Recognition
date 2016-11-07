from hog import HOG
import matplotlib.pyplot as plt

class PedestrianDetector:
	"""docstring for PedestrianDetector"""
	def __init__(self, images):
		self.images = images

	def train(self):
		hog = HOG()
		for image in self.images:
			image_hog = hog.calculate(image)