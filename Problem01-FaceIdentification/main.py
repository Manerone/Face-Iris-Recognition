from faceid_database import YaleFaces
from faceid_database import ORL
from eigenface import Eigenface
import matplotlib.pyplot as plt

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
    print subjects
    recognizer = Eigenface(images, subjects)
    recognizer.train(100)
    correct = 0.0
    for index, subject in enumerate(test_subjects):
        if subject == recognizer.recognize(test_images[index]):
            correct += 1
    accuracy = correct/len(test_subjects)
    print 'Accuracy: ', accuracy * 100, '%'
