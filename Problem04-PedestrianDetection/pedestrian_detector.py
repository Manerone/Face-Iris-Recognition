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
    def __init__(self, images):
        self.images = images
        # self.labels = labels
        self.configurations = {
            'minPyramidSize': (128, 64),
            'windowSize': (128, 64),
            'windowDisplacement': 8,
            'blockSize': (16, 16),
            'blockDisplacement': 8
        }
        self.hog_calculator = HOG()

    def train(self):
        for image in self.images:
            for img in self._pyramidize(image):
                img_ori, img_mag = self._gradient(img)
                a = self._windownize(img_ori)
                b = self._windownize(img_mag)
                for window_ori, window_mag in zip(a, b):
                    start = time.clock()
                    HOG().calculate(window_ori, window_mag)
                    print "Tempo:", (time.clock() - start)

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