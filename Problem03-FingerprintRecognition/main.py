from rindex28_loader import Rindex28Loader
from fingerprint_recognizer import FingerprintRecognizer
from random import randint
import numpy as np


def split_sets(images, labels, n_fingers_to_test=1):
    images = np.array(images)
    labels = np.array(labels)
    training_images = []
    training_labels = []
    test_images = []
    test_labels = []
    size = len(images)
    for index in xrange(0, size, 4):
        random_index = randint(0, 3)
        indexes = range(index + 0, index + 4)
        test_index = indexes[random_index]
        del indexes[random_index]
        for image in images[indexes]:
            training_images.append(image)

        for label in labels[indexes]:
            training_labels.append(label)

        test_images.append(images[test_index])
        test_labels.append(labels[test_index])

    training_images = np.array(training_images)
    training_labels = np.array(training_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return training_images, training_labels, test_images, test_labels


rindex28 = Rindex28Loader('./databases/rindex28')
training_images, training_labels, test_images, test_labels = split_sets(
    rindex28.images, rindex28.subjects
)
recognizer = FingerprintRecognizer(training_images, training_labels)

correct = 0.0
for index, test_image in enumerate(test_images):
    prediction = recognizer.predict(test_image)
    print prediction, test_labels[index]
    if prediction == test_labels[index]:
        correct += 1

print 'Accuracy: ', correct/len(test_images) * 100, '%'
