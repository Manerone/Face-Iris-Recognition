from casia_iris_image_loader import ImageLoaderCASIAIris
from iris_signaturizer import IrisSignaturizer
import scipy.spatial.distance as distance
import platform
import numpy as np
import pandas
from sys import stdout

CLEAR_LINE = '\x1b[2K'


def print_system_info():
    print '=======================SYSTEM INFORMATION=========================='
    print '-> System: ', platform.system(), platform.release()
    print '-> Python Version: ', platform.python_version()
    print '==================================================================='


def print_percent(current, total):
    stdout.write('\r%d%%' % ((current/float(total))*100))
    stdout.flush()


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
    n_of_signatures = len(subjects)
    for i in xrange(n_of_signatures):
        for j in xrange(n_of_signatures):
            if i != j:
                dist = distances[i][j]
                if subjects[i] == subjects[j] and dist > threshold:
                    frr += 1
                elif subjects[i] != subjects[j] and dist <= threshold:
                    far += 1

    frr = (frr/float(n_of_signatures*n_of_signatures)*100)
    far = (far/float(n_of_signatures*n_of_signatures)*100)
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
    minimun = abs(measures[0][1] - measures[0][2])
    eer = 0
    for index, measure in enumerate(measures):
        if minimun > abs(measure[1] - measure[2]):
            minimun = abs(measure[1] - measure[2])
            eer = index
    print('  EER: %.2f %.2f %.2f' % tuple(measures[eer]))


def iris_identification(subjects, normalized_irises):
    print '=======================IDENTIFICATION=============================='

print_system_info()
casia = ImageLoaderCASIAIris('./databases/CASIA-Iris-Lamp-100')
signaturizer = IrisSignaturizer(casia.subjects[:5], casia.images[:5])
signaturizer.generate_signatures()
iris_verification(signaturizer.subjects, signaturizer.signatures)
iris_identification(signaturizer.subjects, signaturizer.normalized_irises)
