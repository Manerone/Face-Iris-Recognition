from rindex28_loader import Rindex28Loader
import numpy as np
import cv2

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
    average_x = (np.sum(np.square(Gx) - np.square(Gy))) / 100
    print average_x
    average_y = np.sum(2 * Gx * Gy) / 100
    angle = np.arctan2(average_x, average_y)
    if average_x < 0 and average_y >= 0:
        angle += np.pi
    elif average_x < 0 and average_y < 0:
        angle -= np.pi
    return (angle / 2)


def orientation_computation(image):
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    orientations = np.array([])
    for i in xrange(30):
        for j in xrange(30):
            Gx = sobelX[i * 10:(i + 1) * 10, j * 10:(j + 1) * 10]
            Gy = sobelY[i * 10:(i + 1) * 10, j * 10:(j + 1) * 10]
            orientations = np.append(orientations, average_gradient(Gx, Gy))
    return orientations


rindex28 = Rindex28Loader('./databases/rindex28')
for image in rindex28.images:
    image_enhanced = image_enhancement(image)
    blurred_image = cv2.medianBlur(image_enhanced, 5)
    orientations = orientation_computation(blurred_image)
