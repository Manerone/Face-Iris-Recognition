from hog import HOG
from pyramid import Pyramid
from windownize import Windownize
from gradient import Gradient

# Extras
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time


class PedestrianDetector:
    """Implementation of a pedestrian detector engine"""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.configurations = {
            'minPyramidSize': (128, 64),
            'windowSize': (128, 64),
            'windowDisplacement': 8,
            'blockSize': (16, 16),
            'blockDisplacement': 8
        }

    def train(self):
        self.hogs = []
        for img in self.images:
            img_ori, img_mag = self._gradient(img)
            start = time.clock()
            hogs.append(HOG().calculate(img_ori, img_mag))
            print "Tempo:", (time.clock() - start)
            # raw_input()

    # def train(self):
    #     for image in self.images:
    #         for img in self._pyramidize(image):
                # img_ori, img_mag = self._gradient(img)
                # a = self._windownize(img_ori)
                # b = self._windownize(img_mag)
                # for window_ori, window_mag in zip(a, b):
                #     start = time.clock()
                #     HOG().calculate(window_ori, window_mag)
                #     print "Tempo:", (time.clock() - start)
                #     raw_input()

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
