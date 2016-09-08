from casia_iris_image_loader import ImageLoaderCASIAIris
from iris_signaturizer import IrisSignaturizer
import scipy.spatial.distance as distance
import platform
import numpy as np
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

    print "FRR:", (frr/float(n_of_signatures*n_of_signatures)*100), '%'
    print "FAR:", (far/float(n_of_signatures*n_of_signatures)*100), '%'


# TODO: transform it into multiprocess
def iris_verification(subjects, signatures):
    distances = calculate_distances(subjects, signatures)
    for threshold in np.arange(0.01, 1, 0.01):
        print "Threshold:", threshold
        verify(subjects, distances, threshold)

print_system_info()
casia = ImageLoaderCASIAIris('./databases/CASIA-Iris-Lamp-100')
signaturizer = IrisSignaturizer(casia.subjects[:500], casia.images[:500])
signaturizer.generate_signatures()
iris_verification(signaturizer.subjects, signaturizer.signatures)
