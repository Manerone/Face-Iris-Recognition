from adjust_gamma import AdjustGamma
from pyramid import Pyramid

class HOG:

	def __init__(self):
		self.configurations = {
			'gamma': 2.0,
			'minPyramidSize': (128, 64)
		}
	
	def calculate(self, image):
		gamma_corrected = AdjustGamma.call(image, self.configurations['gamma'])
		for img in self._pyramidize(image):
			print img.shape
			raw_input("Wait")

	def _pyramidize(self, image):
		return Pyramid.call(
			image, minSize=self.configurations['minPyramidSize']
		)
		
		