from casia_iris_image_loader import ImageLoaderCASIAIris
from iris_signaturizer import IrisSignaturizer
import scipy.spatial.distance as distance
import platform
import numpy as np
import pandas
import matplotlib.pyplot as plt
import random
from sklearn.svm import LinearSVC
from local_binary_patterns import LocalBinaryPatterns


def print_system_info():
    print '=======================SYSTEM INFORMATION=========================='
    print '-> System: ', platform.system(), platform.release()
    print '-> Python Version: ', platform.python_version()
    print '==================================================================='


def showMetrics(metrics):
    x = []
    y = []
    for pt in metrics:
        x.append(pt[1])
        y.append(pt[2])
    plt.axis([0, 100, 0, 100])
    plt.ylabel('FAR')
    plt.xlabel('FRR')
    plt.plot(x, y)
    plt.show()


def calculate_distances(subjects, signatures):
    n_of_signatures = len(signatures)
    distances = np.zeros((n_of_signatures, n_of_signatures))
    for i in xrange(n_of_signatures):
        for j in xrange(n_of_signatures):
            dist = distance.hamming(signatures[i], signatures[j])
            distances[i][j] = dist
    return distances


def verify(subjects, distances, threshold):
    frr = 0
    far = 0
    impostor_scores = 0
    genuine_scores = 0
    n_of_signatures = len(subjects)
    for i in xrange(n_of_signatures):
        for j in xrange(n_of_signatures):
            if i != j:
                dist = distances[i][j]
                if subjects[i] == subjects[j]:
                    genuine_scores += 1
                    if dist > threshold:
                        frr += 1
                elif subjects[i] != subjects[j]:
                    impostor_scores += 1
                    if dist <= threshold:
                        far += 1

    frr = (frr / float(genuine_scores) * 100)
    far = (far / float(impostor_scores) * 100)
    return (frr, far)


def print_far_frr_table(measures):
    padding = np.chararray(len(measures))
    padding[:] = ''
    measures = np.array(measures)
    print 'False Rejection Rates and False Acception Rates\n'
    print pandas.DataFrame(measures, padding, ['Threshold', 'FRR', 'FAR'])
    print ''


def iris_verification(subjects, signatures):
    print '=======================VERIFICATION================================'
    distances = calculate_distances(subjects, signatures)
    measures = []
    for threshold in np.arange(0.05, 1, 0.05):
        frr, far = verify(subjects, distances, threshold)
        measures.append([threshold, frr, far])
    print_far_frr_table(measures)
    showMetrics(measures)
    minimun = abs(measures[0][1] - measures[0][2])
    eer = 0
    for index, measure in enumerate(measures):
        if minimun > abs(measure[1] - measure[2]):
            minimun = abs(measure[1] - measure[2])
            eer = index
    print('  EER: %.2f %.2f %.2f' % tuple(measures[eer]))


def generate_training_and_test_sets(subjects, normalized_irises):
    desc = LocalBinaryPatterns(24, 8)
    training_images = []
    training_subjects = []
    test_images = []
    test_subjects = []
    n_of_subjects = len(subjects)
    size_of_sample = int(n_of_subjects*0.1)
    indexes_to_test = random.sample(xrange(n_of_subjects), size_of_sample)
    for i in xrange(n_of_subjects):
        hist = desc.describe(normalized_irises[i])
        if i in indexes_to_test:
            test_images.append(hist)
            test_subjects.append(subjects[i])
        else:
            training_images.append(hist)
            training_subjects.append(subjects[i])
    training_images = np.array(training_images)
    training_subjects = np.array(training_subjects)
    test_images = np.array(test_images)
    test_subjects = np.array(test_subjects)
    return training_images, training_subjects, test_images, test_subjects


def iris_identification(subjects, normalized_irises):
    print '=======================IDENTIFICATION=============================='
    training_images, training_subjects, test_images, test_subjects = \
        generate_training_and_test_sets(subjects, normalized_irises)
    model = LinearSVC(C=100.0, random_state=42)
    model.fit(training_images, training_subjects)
    predictions = model.predict(test_images)
    accuracy = distance.hamming(test_subjects, predictions)
    print 'Accuracy: ', (1-accuracy)*100, '%'

print_system_info()
casia = ImageLoaderCASIAIris('./databases/CASIA-Iris-Lamp-100')
signaturizer = IrisSignaturizer(casia.subjects[:50], casia.images[:50])
signaturizer.generate_signatures()
# iris_verification(signaturizer.subjects, signaturizer.signatures)
iris_identification(signaturizer.subjects, signaturizer.normalized_irises)
