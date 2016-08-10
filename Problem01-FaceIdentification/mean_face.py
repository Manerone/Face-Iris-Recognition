from faceid_database import YaleFaces
from faceid_database import ORL
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def calculate_mean_face(array_of_images):
    return np.average(array_of_images,axis=0)

def transform_images_to_array(images_in_matrix_form):
	images = []
	for image in images_in_matrix_form:
		images.append(image.flatten())
	return np.array(images)
def images_minus_mean_face(array_of_images, mean_face):
	images = []
	for image in array_of_images:
		images.append(image - mean_face)
	return np.array(images)		

def eigen_faces(images):
	images = np.array(images)
	mean_face = calculate_mean_face(images)
	immf = images_minus_mean_face(images, mean_face)
	images = transform_images_to_array(immf)
	# plt.imshow(mean_face_yale, cmap='Greys_r')
	# plt.show()
	# images_minus_mean_face
	covariance_matrix = np.cov(images)
	eigenvalues, eigenvectors = LA.eig(covariance_matrix)
	print eigenvectors
	print '----'
	print eigenvalues


yale = YaleFaces('./databases/yalefaces/')
print eigen_faces(yale.images)
