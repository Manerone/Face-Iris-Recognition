from rindex28_loader import Rindex28Loader
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

# cv2.imshow('', np.concatenate((sobelX, sobelY), axis=1))
# cv2.waitKey(0)


def image_enhancement(image):
    mean = np.mean(image)
    std = np.std(image)
    image_enhanced = 150 + 95 * ((image - mean) / std)
    image_enhanced[image_enhanced > 255] = 255
    image_enhanced[image_enhanced < 0] = 0
    return np.array(image_enhanced, dtype=np.uint8)


def average_gradient(Gx, Gy):
    average_x = np.sum(np.square(Gx) - np.square(Gy)) / 100
    average_y = np.sum(2 * Gx * Gy) / 100
    return math.atan2(average_y, average_x)/2


def orientation_computation(image):
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    orientations = np.array([])
    for i in xrange(30):
        tmpX = sobelX[i * 10:(i + 1) * 10]
        tmpY = sobelY[i * 10:(i + 1) * 10]
        for j in xrange(30):
            Gx = tmpX[:, j * 10:(j + 1) * 10]
            Gy = tmpY[:, j * 10:(j + 1) * 10]
            orientations = np.append(orientations, average_gradient(Gx, Gy))
    return orientations


def insert_orientation_lines:
    o = 0
    for j in xrange(5, 300, 10):
        for i in xrange(5, 300, 10):
            angle = orientations[o]
            m = np.tan(angle)
            # x0 e y0 são o centro do bloco
            if m > 1:
                # varia x + 4 e x - 4
                # calcula y = m(x-x0) + y0
                pass
            else:
                # varia y + 4 e y - 4
                # calcula x = ((y - y0)/m) + x0
                pass
            f_point = (int(i + 7 * np.cos(angle)), int(j + 7 * np.sin(angle)))
            cv2.line(image, (i, j), f_point, (0, 0, 0))
            o += 1


def region_of_interest_detection(image):
    for i in xrange(30):
        tmpImg = image[i * 10:(i + 1) * 10]
        for j in xrange(30):
            block = tmpImg[:, j * 10:(j + 1) * 10]
            


rindex28 = Rindex28Loader('./databases/rindex28')
for image in rindex28.images:
    image_enhanced = image_enhancement(image)
    blurred_image = cv2.medianBlur(image_enhanced, 5)
    orientations = orientation_computation(blurred_image)
    # interesting_image = region_of_interest_detection(image_enhanced)
    insert_orientation_lines(image, orientations)
    plt.imshow(image, cmap='Greys_r')
    plt.show()
