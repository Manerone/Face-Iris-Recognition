import numpy as np
from numpy import linalg as LA


class Eigenface:

    def __init__(self,images):
        self.images = np.array(images)

    def calculate_mean_face(self, array_of_images):
        mean = np.mean(array_of_images, axis=0, dtype=np.uint32)
        return np.array(mean, dtype=np.uint8)

    def transform_images_to_array(self, images_in_matrix_form):
        images = []
        for image in images_in_matrix_form:
            images.append(image.flatten())
        return np.array(images)

    def images_minus_mean_face(self, array_of_images, mean_face):
        images = []
        for image in array_of_images:
            images.append(image - mean_face)
        return np.array(images)

    def order_eigenvectors_by_eigenvalues(self, eigenvectors, eigenvalues):
        idx = eigenvalues.argsort()[::-1]
        return eigenvectors[:, idx]

    def find_eigenfaces(self, number_of_eigenfaces=5):
        images = self.images # num_imgs X heigth X width
        n_of_images, heigth, width = images.shape

        mean_face = self.calculate_mean_face(images)  # heigth X width
        immf = self.images_minus_mean_face(
            images, mean_face)  # heigth X width

        images = self.transform_images_to_array(
            immf)  # num_img X (number of pixels in images)

        covariance_matrix = np.cov(images)  # num_img x num_img
        eigenvalues, eigenvectors = LA.eig(covariance_matrix)
        eigenvectors = self.order_eigenvectors_by_eigenvalues(
            eigenvectors, eigenvalues)
        eigenvectors = eigenvectors[:number_of_eigenfaces]  # num_img X 1
        eigenvectors = np.real(eigenvectors)

        transposed_images = images.transpose() #  (number of pixels in images) X num_img
        self.eigenfaces = []
        for eigenvector in eigenvectors:
            multiplication = np.dot(transposed_images, eigenvector)
            self.eigenfaces.append(multiplication.reshape(heigth, width))
        self.eigenfaces = np.array(self.eigenfaces)
        return self.eigenfaces
