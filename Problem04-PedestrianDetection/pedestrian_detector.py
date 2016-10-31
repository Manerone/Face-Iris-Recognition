from adjust_gamma import AdjustGamma
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
			plt.imshow(image)
			plt.show()

			gamma_corrected = AdjustGamma.call(image, self.configurations['gamma'])
			plt.imshow(gamma_corrected)
			plt.show()
		