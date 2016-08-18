import numpy as np
from numpy import linalg as LA


class Eigenface:

    # Images: Array of images that will be the database
    def __init__(self, images):
        im = np.array(images)
        n_images, height, width = im.shape
        self.n_images = n_images
        self.image_height = height
        self.image_width = width
        self.images = self.transform_images_to_array(im)
        self.projected_images = None
        self.mean_face = None
        self.eigenfaces = None

    def get_image(self, index):
        return self.images[index].reshape(self.image_height, self.image_width)

    def reconstruct_image(self, index):
        projected_image = self.projected_images[index]
        eigenfaces = self.eigenfaces
        multiplication = np.dot(eigenfaces.transpose(), projected_image)
        result = np.add(multiplication, self.mean_face).reshape(
            self.image_height, self.image_width)
        return result

    def calculate_mean_face(self, array_of_images):
        return np.average(array_of_images, axis=0)

    def transform_images_to_array(self, images_in_matrix_form):
        length, height, width = images_in_matrix_form.shape
        return images_in_matrix_form.reshape(length, height*width)

    def images_minus_mean_face(self, array_of_images):
        return array_of_images - self.get_mean_face()

    def order_eigenvectors_by_eigenvalues(self, eigenvectors, eigenvalues):
        idx = eigenvalues.argsort()[::-1]
        return eigenvectors[idx]

    def find_eigenfaces(self, array_of_images, number_of_eigenfaces=5):

        # num_imgs X (number of pixels in images)
        images = self.images_minus_mean_face(array_of_images)

        # num_img x num_img
        covariance_matrix = np.dot(images, images.T)
        eigenvalues, eigenvectors = LA.eig(covariance_matrix)
        eigenvectors = eigenvectors.transpose()
        eigenvectors = self.order_eigenvectors_by_eigenvalues(
            eigenvectors, eigenvalues)
        # Each eigenvector has num_img X 1
        eigenvectors = eigenvectors[:number_of_eigenfaces]
        eigenvectors = np.real(eigenvectors)

        # (number of pixels in images) X num_img
        transposed_images = images.transpose()
        eigenfaces = []
        for eigenvector in eigenvectors:
            # result is an array with (number of pixels in images) size
            result = np.dot(transposed_images, eigenvector)
            eigenfaces.append(result/LA.norm(result))
        eigenfaces = np.array(eigenfaces)
        # n_eigenfaces X (number of pixels in images)
        self.eigenfaces = eigenfaces
        return eigenfaces

    def get_mean_face(self):
        if self.mean_face is None:
            self.mean_face = self.calculate_mean_face(self.images)
        return self.mean_face

    def project_image(self, image, eigenfaces, mean_face):
        # (number of pixels in images) X number_of_eigenfaces *
        # (number of pixels in images) X 1
        return np.dot(eigenfaces, (image - mean_face).transpose())

    def project_images(self, images, eigenfaces, mean_face):
        tmp = []
        for image in images:
            tmp.append(
                self.project_image(image, eigenfaces, mean_face))
        self.projected_images = np.array(tmp)

    def train(self, number_of_eigenfaces=5):
        eigenfaces = self.find_eigenfaces(self.images, number_of_eigenfaces)
        images = self.images
        mean_face = self.get_mean_face()
        self.project_images(images, eigenfaces, mean_face)
