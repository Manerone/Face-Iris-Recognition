import numpy as np
import matplotlib.pyplot as plt
import math


class Gradient:

    def call(self, image):
        hor_grad = self._horizontal_gradient(image)
        ver_grad = self._vertical_gradient(image)
        return self._orientation_and_magnitude(hor_grad, ver_grad)

    def _horizontal_gradient(self, image):
        heigth, width, channels = image.shape
        splitted_channels = self._split(image)
        grad = np.array([np.zeros((heigth, width)) for i in xrange(channels)])
        for channel in xrange(channels):
            tmp = splitted_channels[channel]
            for i in xrange(heigth):
                for j in xrange(1, width - 1):
                    grad[channel][i][j] = tmp[i][j-1] - tmp[i][j+1]
        return grad

    def _vertical_gradient(self, image):
        heigth, width, channels = image.shape
        splitted_channels = self._split(image)
        grad = np.array([np.zeros((heigth, width)) for i in xrange(channels)])
        for channel in xrange(channels):
            tmp = splitted_channels[channel]
            for i in xrange(1, heigth - 1):
                for j in xrange(width):
                    grad[channel][i][j] = tmp[i-1][j] - tmp[i+1][j]
        return grad

    def _orientation_and_magnitude(self, horizontal, vertical):
        channels, heigth, width = horizontal.shape
        orientations = np.zeros((heigth, width))
        magnitudes = np.zeros((heigth, width))
        for i in xrange(heigth):
            for j in xrange(width):
                mag = []
                ori = []
                for channel in xrange(channels):
                    mag.append(
                        self._magnitude(
                            horizontal[channel][i][j],
                            vertical[channel][i][j]
                        )
                    )
                    ori.append(
                        self._orientation(
                            horizontal[channel][i][j],
                            vertical[channel][i][j]
                        )
                    )
                    m = max(mag)
                    magnitudes[i][j] = m
                    orientations[i][j] = ori[mag.index(m)]
        return orientations, magnitudes

    def _split(self, image):
        red = image[:, :, 2]
        green = image[:, :, 1]
        blue = image[:, :, 0]
        return red, green, blue

    def _magnitude(self, value1, value2):
        return math.sqrt(value1 * value1 + value2 * value2)

    def _orientation(self, x, y):
        return math.atan2(y, x)
