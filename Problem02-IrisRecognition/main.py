from casia_iris_image_loader import ImageLoaderCASIAIris
from iris_recognizer import IrisRecognizer


casia = ImageLoaderCASIAIris('./databases/CASIA-Iris-Lamp-100')
recognizer = IrisRecognizer(casia.subjects, casia.images, acceptance_threshold=0.8)
recognizer.train()
