from faceid_database import YaleFaces
from faceid_database import ORL
from eigenface import Eigenface
import numpy as np
import random
import platform
from sys import stdout


CLEAR_LINE = '\x1b[2K'


def print_system_info():
    print '-----------------------SYSTEM INFORMATION--------------------------'
    print '-> System: ', platform.system(), platform.release()
    print '-> Python Version: ', platform.python_version()


def calculate_yale():
    print '-----------------------Yale Faces Tests----------------------------'
    yale = YaleFaces('./databases/yalefaces/')
    for label in yale.class_labels:
        print 'Testing remotion of label: ', label
        images = []
        subjects = []
        test_images = []
        test_subjects = []
        for index, val in enumerate(yale.classes):
            if val == label:
                test_subjects.append(yale.subjects[index])
                test_images.append(yale.images[index])
            else:
                subjects.append(yale.subjects[index])
                images.append(yale.images[index])
        recognizer = Eigenface(images, subjects)
        recognizer.train(60)
        correct = 0
        for index, test_subject in enumerate(test_subjects):
            result = recognizer.recognize(test_images[index])
            if test_subject == result:
                correct += 1
        accuracy = correct / float(len(test_subjects))
        print 'Accuracy: ', accuracy * 100, '%'


def calculate_orl():
    print '-----------------------ORL Faces Tests-----------------------------'
    number_of_eigenfaces_on_each_test = [5, 10, 50, 100, 200, 300]
    orl = ORL('./databases/orl_faces/')
    n_of_images = len(orl.images)
    k_fold = 10
    n_of_itens_per_fold = n_of_images / k_fold
    for n_of_eigenfaces in number_of_eigenfaces_on_each_test:
        print 'Testing with', n_of_eigenfaces, 'eigenfaces'
        means = []
        for k in xrange(k_fold):
            stdout.write('\r%d%%' % ((k/float(k_fold))*100))
            stdout.flush()
            images = []
            subjects = []
            test_images = []
            test_subjects = []
            n_of_correct_guesses = 0
            n_of_tries = 0
            indexes_to_test = random.sample(
                range(n_of_images), n_of_itens_per_fold)
            for index in xrange(n_of_images):
                if index in indexes_to_test:
                    test_images.append(orl.images[index])
                    test_subjects.append(orl.subjects[index])
                else:
                    images.append(orl.images[index])
                    subjects.append(orl.subjects[index])

            recognizer = Eigenface(images, subjects)
            recognizer.train(n_of_eigenfaces)

            for index, test_subject in enumerate(test_subjects):
                n_of_tries += 1
                result = recognizer.recognize(test_images[index])
                if test_subject == result:
                    n_of_correct_guesses += 1
            means.append(n_of_correct_guesses / float(n_of_tries))
        stdout.write('\r' + CLEAR_LINE)
        print 'Accuracy:', np.mean(means)*100, '%'
        print 'Standart Deviation:', np.std(means), '\n'


# MAIN
print_system_info()
calculate_yale()
calculate_orl()
