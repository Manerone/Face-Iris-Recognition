# Filtro de canny (python canny)
# ler todas as images da base
# preprocessar as imagens
# aplicar o filtro de canny em cada uma delas
# aplicar a transformada circular hough (tentar implementar ela)
# com isso eu detectei a pupila
# dado o centro da pupila conseguimos detectar a iris
# http://stackoverflow.com/questions/9860667/writing-robust-color-and-size-invariant-circle-detection-with-opencv-based-on
# http://stackoverflow.com/questions/10716464/what-are-the-correct-usage-parameter-values-for-houghcircles-in-opencv-for-iris
import cv2
from casia_iris_image_loader import ImageLoaderCASIAIris
import numpy as np
# from PIL import Image
from matplotlib import pyplot as plt


def show_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def find_circles(image):
    accumulator_threshold = 70
    circles = None
    while circles is None:
        circles = cv2.HoughCircles(image, cv2.cv.CV_HOUGH_GRADIENT,
                                   1, 1, param2=accumulator_threshold)
        accumulator_threshold = accumulator_threshold - 1
    circles = np.uint16(np.around(circles))
    return circles[0][0]


def pre_process_img(image):
    img = image
    img_with_blur = cv2.medianBlur(img, 15)
    edges = cv2.Canny(img_with_blur, 50, 200)
    # show_img(edges)
    return edges


casia = ImageLoaderCASIAIris('./databases/CASIA-Iris-Lamp-100')
for index, image in enumerate(casia.images):
    print casia.subjects[index]
    processed_img = pre_process_img(image)
    circles = find_circles(processed_img)
    cv2.circle(image, (circles[0], circles[1]), circles[2], (0, 255, 0), 1)
    cv2.circle(image, (circles[0], circles[1]), 2, (0, 0, 255), 3)
    show_img(image)
# img = Image.open(
#     './databases/CASIA-Iris-Lamp-100/085/R/S2085R02.jpg').convert('L')
# img = np.array(img, 'uint8')
# processed_img = pre_process_img(img)
# circles = find_circles(processed_img)
# cv2.circle(img, (circles[0], circles[1]), circles[2], (0, 255, 0), 1)
# cv2.circle(img, (circles[0], circles[1]), 2, (0, 0, 255), 3)
# show_img(img)
