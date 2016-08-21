from faceid_database import YaleFaces
from faceid_database import ORL
from eigenface import Eigenface


def calculate_yale():
    print 'Testing YaleFaces database'
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
        accuracy = correct/float(len(test_subjects))
        print 'Accuracy: ', accuracy * 100, '%'


def calculate_orl():
    print 'Testing ORL database'
    orl = ORL('./databases/orl_faces/')
    n_of_images = len(orl.images)
    k_fold = 10
    n_of_itens_per_fold = n_of_images/k_fold
    n_of_correct_guesses = 0
    n_of_tries = 0
    for k in xrange(k_fold):
        print 'Calculating', k+1, 'subset of', k_fold, 'subsets'
        images = []
        subjects = []
        test_images = []
        test_subjects = []
        for index, image in enumerate(orl.images):
            if k*n_of_itens_per_fold <= index < (k+1) * n_of_itens_per_fold:
                test_images.append(image)
                test_subjects.append(orl.subjects[index])
            else:
                images.append(image)
                subjects.append(orl.subjects[index])
        recognizer = Eigenface(images, subjects)
        recognizer.train(150)
        for index, test_subject in enumerate(test_subjects):
            n_of_tries += 1
            if test_subject == recognizer.recognize(test_images[index]):
                n_of_correct_guesses += 1
    accuracy = n_of_correct_guesses/float(n_of_tries)
    print 'Accuracy: ', accuracy * 100, '%'


# MAIN
calculate_yale()
print '----------------------------------------------------------------------'
calculate_orl()
