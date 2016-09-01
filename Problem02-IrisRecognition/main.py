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
from PIL import Image
from operator import itemgetter


def show_img(imgs):
    cv2.imshow("output", np.hstack(imgs))
    cv2.waitKey(0)


def find_pupil(image):
    accumulator_threshold = 150
    circles = None
    while circles is None:
        circles = cv2.HoughCircles(image, cv2.cv.CV_HOUGH_GRADIENT,
                                   1, 1, param2=accumulator_threshold)
        accumulator_threshold = accumulator_threshold - 1
    circles = np.uint16(np.around(circles[0, :]))
    return circles[0]


def get_points_near_circle_perimeter(x, y, r, img, num_of_points=360):
    points = []
    distance_of_points = (2 * np.pi) / num_of_points
    for i in range(num_of_points):
        teta = i * distance_of_points
        y_p = y + int(r * np.cos(teta))
        x_p = x + int(r * np.sin(teta))
        points.append(img[y_p][x_p])
    return np.array(points)


def find_iris((x_pupil, y_pupil, r_pupil), img, value=4):
    image = cv2.equalizeHist(img)
    n_of_rows, n_of_columns = img.shape
    max_r = min((n_of_rows - y_pupil, n_of_columns - x_pupil))
    x, y, r_b = x_pupil, y_pupil, r_pupil
    r_a = r_b + 5
    variations = []
    points_before = get_points_near_circle_perimeter(x, y, r_b, image)
    iterations = 0
    max_iterations = 30
    while r_a < max_r and iterations < max_iterations:
        points_after = get_points_near_circle_perimeter(x, y, r_a, image)
        variations.append((r_a, sum(abs(points_after - points_before))))
        points_before = points_after
        r_a += 5
        iterations += 1
    return x, y, max(variations, key=itemgetter(1))[0]


def pre_process_img(image):
    # Threshold.
    # Set values equal to or above 35 to 255.
    # Set values below 35 to 0.

    img = image
    img_with_blur = cv2.medianBlur(img, 25)
    th, im_th = cv2.threshold(img_with_blur, 25, 255, cv2.THRESH_BINARY)

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (w / 2, h - 1), 0)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th & im_floodfill_inv
    # cv2.imshow("Original Image", image)
    # cv2.imshow("Thresholded Image", im_th)
    # cv2.imshow("Floodfilled Image", im_floodfill)
    # cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    # cv2.imshow("Foreground", im_out)
    # cv2.waitKey(0)
    canny = cv2.Canny(im_out, 50, 200)
    return canny


def normalize_iris((x_pupil, y_pupil, r_pupil), (x_iris, y_iris, r_iris),
                   image, n_of_divisions=360):
    distance_of_each_division = (2*np.pi)/n_of_divisions
    normalized_image = []
    for division in xrange(n_of_divisions):
        teta = distance_of_each_division * division
        points = []
        for radius in xrange(int(r_pupil), int(r_iris+1)):
            y_p = y_iris + int(radius * np.cos(teta))
            x_p = x_iris + int(radius * np.sin(teta))
            points.append(image[y_p][x_p])
        normalized_image.append(points)
    normalized_image = np.array(normalized_image).T
    return cv2.resize(normalized_image, (600, 100), interpolation=cv2.INTER_CUBIC)

casia = ImageLoaderCASIAIris('./databases/CASIA-Iris-Lamp-100')
for index, image in enumerate(casia.images):
    print "Preprocessing image"
    processed_img = pre_process_img(image)
    print "Finding pupil"
    x_pupil, y_pupil, r_pupil = find_pupil(processed_img)
    pupil_coords = (x_pupil, y_pupil, r_pupil)
    print "Finding iris"
    x_iris, y_iris, r_iris = find_iris(pupil_coords, image)
    iris_coords = (x_iris, y_iris, r_iris)
    print "Segmenting image"
    normalized_iris = normalize_iris(pupil_coords, iris_coords, image)

    cv2.circle(image, (x_pupil, y_pupil), r_pupil, (0, 255, 0), 1)
    cv2.circle(image, (x_pupil, y_pupil), 2, (0, 0, 255), 3)

    cv2.circle(image, (x_iris, y_iris), r_iris, (255, 255, 0), 1)
    cv2.circle(image, (x_iris, y_iris), 2, (0, 0, 255), 3)

    show_img([processed_img, image])
    show_img([normalized_iris])
# image = Image.open(
#     './databases/CASIA-Iris-Lamp-100/097/L/S2097L03.jpg').convert('L')
# image = np.array(image, 'uint8')
# processed_img = pre_process_img(image)
# x_pupil, y_pupil, r_pupil = find_pupil(processed_img)
# pupil_coords = (x_pupil, y_pupil, r_pupil)
#
# x_iris, y_iris, r_iris = find_iris(pupil_coords, image)
# iris_coords = (x_iris, y_iris, r_iris)
#
# normalized_iris = normalize_iris(pupil_coords, iris_coords, image)
#
# cv2.circle(image, (x_pupil, y_pupil), r_pupil, (0, 255, 0), 1)
# cv2.circle(image, (x_pupil, y_pupil), 2, (0, 0, 255), 3)
#
# cv2.circle(image, (x_iris, y_iris), r_iris, (255, 255, 0), 1)
# cv2.circle(image, (x_iris, y_iris), 2, (0, 0, 255), 3)
#
# show_img([normalized_iris])
# show_img([processed_img, image])
