import numpy as np
from numpy import linalg as LA


class Eigenface:

    # Eigenface constructor
    # Params:
    #   +Images+ - Array of images that will be the database
    #         shape: (num_img, height, widht)
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

    # @@@@@@@@@@@@@@@@@@@@@@@@@ PUBLIC INTERFACE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Only this methods should be used outside the class, the others should be
    # private, but I was to lazy to find how to do it.

    # Creates all the necessary things for image reconstruction and recognition
    # Param:
    #   +number_of_eigenfaces+ - Option parameters, how many eigenfaces should
    #       be calculated.
    # Obeservations:
    #   - If you want to change the number of the eigenfaces call train method
    #       again with the number you desire.
    def train(self, number_of_eigenfaces=5):
        eigenfaces = self.find_eigenfaces(self.images, number_of_eigenfaces)
        images = self.images
        mean_face = self.get_mean_face()
        self.project_images(images, eigenfaces, mean_face)

    # Get image by index
    # Params:
    #   +index+ - integer to find image
    # Return: Original image provided on the constructor
    def get_image(self, index):
        return self.images[index].reshape(self.image_height, self.image_width)

    def reconstruct_image(self, index):
        projected_image = self.projected_images[index].reshape(-1, 1)
        eigenfaces = self.eigenfaces
        mean_face = self.get_mean_face().reshape(-1, 1)
        multiplication = np.dot(eigenfaces.transpose(), projected_image)
        result = np.add(multiplication, mean_face).reshape(
            self.image_height, self.image_width)
        return result

    # @@@@@@@@@@@@@@@@@@@@@@@ END PUBLIC INTERFACE @@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # Calculate the average face based on the provided array of images
    # Params:
    #   +array_of_images+ - Array of images to find the average face.
    # Obeservations:
    #   - every image in the array_of_images should be in the array format,
    #       see transform_images_to_array method for more information.
    # Return: The average face with the shape (1, height * width)
    def calculate_mean_face(self, array_of_images):
        return np.average(array_of_images, axis=0).reshape(1, -1)

    # Receives an array of images in a matrix pixel format and trasnforms it
    # into an array of images in array format
    # Params:
    #   +images_in_matrix_form+ - Array of images(matrix format) that will be
    #       transformed into an array of images(array format)
    # Return: An array with shape (num_imgs, height * width)
    def transform_images_to_array(self, images_in_matrix_form):
        length, height, width = images_in_matrix_form.shape
        return images_in_matrix_form.reshape(length, height * width)

    # Returns the difference between each image of array_of_images and the
    #   average face
    # Params:
    #   +array_of_images+ - Array of images to subtract the average face
    # Obeservations:
    #   - every image in the array_of_images should be in the array format,
    #       see transform_images_to_array method for more information.
    # Return: Difference between every image and the average face
    def images_minus_mean_face(self, array_of_images):
        return array_of_images - self.get_mean_face()

    # Returns the eigenvectors ordered descending by their eigenvalue
    # Params:
    #   +eigenvectors+ - Array of eigenvectors, each row of the array is an
    #       eigenvector
    #   +eigenvalues+ - Eigenvalue of their corresponding eigenvalue
    # Obeservations:
    #   - The relation between the eigenvector and the eigenvalue is
    #       by the index, the first eigenvalue is correspondent to the first
    #       eigenvector, and so on.
    # Return: Eigenvectors ordered by ther eigenvalues,
    #         shape (num_of_eigenvectors, lenght_of_eigenvector)
    def order_eigenvectors_by_eigenvalues(self, eigenvectors, eigenvalues):
        idx = eigenvalues.argsort()[::-1]
        return eigenvectors[idx]

    # Finds the eigenvectors of matrix, ordered by their eigenvalue and in
    # real format
    # Params:
    #   +matrix+ - Matrix to find the eigenvectors
    # Return: Eigenvectors of matrix, ordered by their eigenvalue and in
    # real format
    def find_eigenvectors(self, matrix):
        covariance_matrix = np.dot(matrix, matrix.T)
        eigenvalues, eigenvectors = LA.eig(covariance_matrix)
        eigenvectors = self.order_eigenvectors_by_eigenvalues(
            eigenvectors.transpose(), eigenvalues)
        # Each eigenvector has num_img X 1
        return np.real(eigenvectors)

    # Calculate the eigenfaces based on images and eigenvectors
    # Params:
    #   +images+ - Matrix of images, the images should be in the rows
    #       of the matrix
    #   +eigenvectors+ - Matrix of eigenvectors, the eigenvectors should be in
    #       the rows of the matrix
    # Obeservations:
    #   - Images should be in the array format,
    #     see transform_images_to_array method for more information.
    def calculate_eigenfaces(self, images, eigenvectors):
        # (number of pixels in images) X num_img
        transposed_images = images.transpose()
        eigenfaces = []
        for eigenvector in eigenvectors:
            # result is an array with (number of pixels in images) size
            result = np.dot(transposed_images, eigenvector)
            eigenfaces.append(result / LA.norm(result))
        return np.array(eigenfaces)

    # Returns the eigenfaces of the array_of_images
    # Params:
    #   +array_of_images+ - Images to find the eigenfaces
    #   +number_of_eigenfaces+ - Optional param, number of eigenfaces to be
    #       generated
    # Obeservations:
    #   - For more information on the array format see the method
    #       transform_images_to_array.
    # Return: Eigenfaces in the array format
    def find_eigenfaces(self, array_of_images, number_of_eigenfaces=5):
        # images has shape of (num_imgs, height * width)
        images = self.images_minus_mean_face(array_of_images)
        # eigenvectors has shape of (number_of_eigenfaces, num_imgs)
        eigenvectors = self.find_eigenvectors(images)[:number_of_eigenfaces]
        # eigenfaces has shape of (number_of_eigenfaces, height * width)
        self.eigenfaces = self.calculate_eigenfaces(images, eigenvectors)
        return self.eigenfaces

    # Singleton pattern, enforces that the mean_face is only calculated once
    # Return: The mean face of the provided faces in the constructor
    #   with the shape (1, width * height)
    def get_mean_face(self):
        if self.mean_face is None:
            self.mean_face = self.calculate_mean_face(self.images)
        return self.mean_face

    # Projects an image into the face space
    # Params:
    #   +image+ - Image to be projected into the face space
    #   +eigenfaces+ - eigenfaces of the face space
    #   +mean_face+ - average face of the face space
    # Obeservations:
    #   - Image and mean_face should be in the array format,
    #       the shape should be (1,height*width)
    #   - Eigenfaces should be in array format, each eigenface should be
    #       an array represent as a row.
    # Return: An array of shape(1, number_of_eigenfaces)
    def project_image(self, image, eigenfaces, mean_face):
        # (number of pixels in images) X number_of_eigenfaces *
        # (number of pixels in images) X 1
        return np.dot(eigenfaces, (image - mean_face).transpose())

    # Creates the projections on the face space of the provided images
    # Params:
    #   +images+ - Array of images in the array format.
    #   +eigenfaces+ - Eigenfaces of the face space
    #   +mean_face+ - Average face of the face space
    # Obeservations:
    #   - For more information on the array format see the
    #       transform_images_to_array method.
    # Return: nothing
    def project_images(self, images, eigenfaces, mean_face):
        tmp = []
        for image in images:
            tmp.append(
                self.project_image(image, eigenfaces, mean_face))
        self.projected_images = np.array(tmp)
