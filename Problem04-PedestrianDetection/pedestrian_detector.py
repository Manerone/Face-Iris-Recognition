from hog import HOG
from pyramid import Pyramid
from windownize import Windownize
from gradient import Gradient
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


class PedestrianDetector:
    """docstring for PedestrianDetector"""
    def __init__(self, images):
        self.images = images
        self.configurations = {
            'minPyramidSize': (128, 64),
            'windowSize': (128, 64),
            'windowDisplacement': 8
        }

    def train(self):
        hog = HOG()
        for image in self.images:
            for img in self._pyramidize(image):
                img_ori, img_mag = self._gradient(img)
                a = self._windownize(img_ori)
                b = self._windownize(img_mag)
                for window_ori, window_mag in zip(a, b):
                    # self._show(window_ori, window_mag)
                    pass

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

    def _gradient(self, image):
        return Gradient().call(image)

    def _show(self, orientations, magnitudes):
        heigth, width = orientations.shape
        img = np.ones((heigth, width))
        img.fill(255)
        for i in xrange(heigth):
            for j in xrange(width):
                x = (int)(j + magnitudes[i][j] * math.cos(orientations[i][j]))
                y = (int)(i + magnitudes[i][j] * math.sin(orientations[i][j]))
                point2 = (x, y)
                cv2.line(img, (j, i), point2, (0, 0, 0))
        plt.imshow(img, cmap='Greys')
        plt.show()
