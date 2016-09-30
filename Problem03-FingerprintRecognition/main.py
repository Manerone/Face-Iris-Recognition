import math
import numpy as np
from matplotlib import pyplot as plt
import cv2
import copy
from sklearn.preprocessing import normalize
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
    return np.reshape(orientations, (30, 30)) * -1


def show_orientation_lines(image_org, orientations):
    image = copy.deepcopy(image_org)
    lin_block = 0
    for lin in xrange(5, 300, 10):
        col_block = 0
        for col in xrange(5, 300, 10):
            angle = orientations[lin_block][col_block]
            n = np.tan(angle)
            if n == 0:
                s_point = (col + 4, lin)
                f_point = (col - 4, lin)
            elif(np.abs(n) > 1):
                # varia y + 4 e y - 4
                # calcula x = ((y - y0)/n) + x0
                s_point = (col - 4, int((col - 4 - col)/n + lin))
                f_point = (col + 4, int((col + 4 - col)/n + lin))
            else:
                # varia x + 4 e x - 4
                # calcula y = n(x-x0) + y0
                s_point = (int(n*(lin - 4 - lin) + col), lin - 4)
                f_point = (int(n*(lin + 4 - lin) + col), lin + 4)
            cv2.line(image, s_point, f_point, (0, 0, 0))
            col_block += 1
        lin_block += 1
    plt.imshow(image, cmap='Greys_r')
    plt.show()


def show_interesting_blocks(image_original, interesting_blocks):
    image = copy.deepcopy(image_original)
    lin_block = 0
    for lin in xrange(5, 300, 10):
        col_block = 0
        for col in xrange(5, 300, 10):
            if interesting_blocks[lin_block][col_block]:
                cv2.circle(image, (col, lin), 2, (0, 0, 0), -1)
            col_block += 1
        lin_block += 1
    plt.imshow(image, cmap='Greys_r')
    plt.show()


def regions_of_interest(image):
    interesting_blocks = np.zeros((30, 30), dtype=np.bool)
    max_distance = 150 * np.sqrt(2)
    for i in xrange(30):
        tmpImg = image[i * 10:(i + 1) * 10]
        for j in xrange(30):
            block = tmpImg[:, j * 10:(j + 1) * 10]
            curret_distance = np.linalg.norm([150 - i, 150 - j])
            distance_ratio = (max_distance - curret_distance)/max_distance
            mean = np.mean(block)/255.0
            standard_deviation = np.std(block)/255.0
            v = 0.5 * (1-mean) + 0.5 * standard_deviation + distance_ratio
            if v > 0.3:
                interesting_blocks[i][j] = True
    return interesting_blocks


rindex28 = Rindex28Loader('./databases/rindex28')
for image in rindex28.images:
    image_enhanced = image_enhancement(image)
    blurred_image = cv2.medianBlur(image_enhanced, 5)
    orientations = orientation_computation(blurred_image)
    show_orientation_lines(image, orientations)
    interesting_blocks = regions_of_interest(image_enhanced)
    show_interesting_blocks(image, interesting_blocks)
