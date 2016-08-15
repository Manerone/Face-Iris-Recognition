import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

class Eigenface:
	@staticmethod
	def calculate_mean_face(array_of_images):
		return np.average(array_of_images,axis=0)

	@staticmethod
	def transform_images_to_array(images_in_matrix_form):
		images = []
		for image in images_in_matrix_form:
			images.append(image.flatten())
		return np.array(images)

	@staticmethod
	def images_minus_mean_face(array_of_images, mean_face):
		images = []
		for image in array_of_images:
			images.append(image - mean_face)
		return np.array(images)	

	@staticmethod
	def calculate(images, number_of_eigenfaces = 5):
		images = np.array(images)
		mean_face = Eigenface.calculate_mean_face(images)
		immf = Eigenface.images_minus_mean_face(images, mean_face)
		images = Eigenface.transform_images_to_array(immf)
		covariance_matrix = np.cov(images)
		eigenvalues, eigenvectors = LA.eig(covariance_matrix)
		eigenvectors = eigenvectors[:number_of_eigenfaces]
		print eigenvectors
		
		