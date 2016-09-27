import math
import numpy as np
from matplotlib import pyplot as plt
import cv2
import copy
from rindex28_loader import Rindex28Loader


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
    return orientations * -1


def insert_orientation_lines(image_org, orientations):
    image = copy.deepcopy(image_org)
    o = 0
    for lin in xrange(5, 300, 10):
        for col in xrange(5, 300, 10):
            angle = orientations[o]
            n = np.tan(angle)
            print angle*180/np.pi, n
            if n == 0:
                s_point = (col + 4, lin)
                f_point = (col - 4, lin)
                cv2.line(image, s_point, f_point, (0, 0, 0))
            elif(np.abs(n) > 1):
                print 'find X'
                # varia y + 4 e y - 4
                # calcula x = ((y - y0)/n) + x0
                s_point = (col - 4, int((col - 4 - col)/n + lin))
                f_point = (col + 4, int((col + 4 - col)/n + lin))
                cv2.line(image, s_point, f_point, (0, 0, 0))
                # plt.imshow(image, cmap='Greys_r')
                # plt.show()
            else:
                print 'find Y'
                # varia x + 4 e x - 4
                # calcula y = n(x-x0) + y0
                s_point = (int(n*(lin - 4 - lin) + col), lin - 4)
                f_point = (int(n*(lin + 4 - lin) + col), lin + 4)
                cv2.line(image, s_point, f_point, (0, 0, 0))
                # plt.imshow(image, cmap='Greys_r')
                # plt.show()
            # cv2.line(image, s_point, f_point, (0, 0, 0))
            o += 1
    plt.imshow(image, cmap='Greys_r')
    plt.show()


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
    insert_orientation_lines(image, orientations)
    # interesting_image = region_of_interest_detection(image_enhanced)
