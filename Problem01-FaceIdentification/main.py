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
    recognizer = Eigenface(images, subjects)
    recognizer.train(156)
    correct = 0
    for index, subject in enumerate(test_subjects):
        result = recognizer.recognize(test_images[index])
        if subject == result:
            correct += 1
    accuracy = correct/float(len(test_subjects))
    print 'Accuracy: ', accuracy * 100, '%'
