import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal


class Gradient:

    def call(self, image):
        grad_x = self._gradient_x(image)
        grad_y = self._gradient_y(image)
        return self._orientation_and_magnitude(grad_x, grad_y)

    def _gradient_x(self, image):
        grad = []
        filt = np.array([[-1, 0, 1]])
        for channel in self._split(image):
            grad.append(signal.convolve2d(channel, filt, mode='same'))
        return np.array(grad)

    def _gradient_y(self, image):
        grad = []
        filt = np.array([[-1], [0], [1]])
        for channel in self._split(image):
            grad.append(signal.convolve2d(channel, filt, mode='same'))
        return np.array(grad)

    def _orientation_and_magnitude(self, grad_x, grad_y):
        channels, heigth, width = grad_x.shape
        orientations = np.zeros((heigth, width))
        magnitudes = np.zeros((heigth, width))
        for i in xrange(heigth):
            for j in xrange(width):
                mag = []
                ori = []
                for channel in xrange(channels):
                    mag.append(
                        self._magnitude(
                            grad_x[channel][i][j],
                            grad_y[channel][i][j]
                        )
                    )
                    ori.append(
                        self._orientation(
                            grad_x[channel][i][j],
                            grad_y[channel][i][j]
                        )
                    )
                m = max(mag)
                magnitudes[i][j] = m
                orientations[i][j] = ori[mag.index(m)]
        return np.absolute(orientations), magnitudes

    def _split(self, image):
        red = image[:, :, 2]
        green = image[:, :, 1]
        blue = image[:, :, 0]
        return red, green, blue

    def _magnitude(self, value1, value2):
        return math.sqrt(value1 * value1 + value2 * value2)

    def _orientation(self, x, y):
        return math.atan2(y, x) % math.pi
