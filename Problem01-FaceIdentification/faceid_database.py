#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image

#class ORL:

class Database:

	_path = '' ## virtual path
	def __init__(self, path = _path):
		self.path = path
		self.get_images_and_labels(self.path)

class ORL(Database):

	_path = './orl_faces'

	def get_images_and_labels(self,path = _path):
		# images will contains face images
		self.images = []
		# subjets will contains the subject identification number assigned to the image
		self.subjects = []

		subjects_paths = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
		for s,subject_paths in enumerate(subjects_paths, start=1):
			subject_path = [os.path.join(subject_paths, f) for f in os.listdir(subject_paths) if f.endswith('.pgm') and os.path.isfile(os.path.join(subject_paths,f)) ]

			for image_path in subject_path:
#				print 'sub: {0}'.format(image_path)
				# Read the image and convert to grayscale
				image_pil = Image.open(image_path).convert('L')
				# Convert the image format into numpy array
				image = np.array(image_pil, 'uint8')
				# Get the label of the image
				nbr = int(os.path.split(image_path)[1].split(".")[0])

				self.images.append(image)
				self.subjects.append(nbr)

			print 'sub: {0}({1}#) - {2}'.format(s,len(subject_path),subject_paths)


class YaleFaces(Database):

	_path = './yalefaces'
	# classes: center-light, w/glasses, happy, left-light, w/no glasses, normal, right-light, sad, sleepy, surprised, and wink.
	class_labels = ['.centerlight', '.glasses', '.happy', '.leftlight', '.noglasses', '.normal', '.rightlight', '.sad', '.sleepy', '.surprised','.wink']
	# Note that the image "subject04.sad" has been corrupted and has been substituted by "subject04.normal".
	# Note that the image "subject01.gif" corresponds to "subject01.centerlight" :~ mv subject01.gif subject01.centerlight


	def get_images_and_labels(self,path = _path):
		# Append all the absolute image paths in a list image_paths
		# We will not read the image with the .sad extension in the training set
		# Rather, we will use them to test our accuracy of the training

		# images will contains face images
		self.images = []
		# subjets will contains the subject identification number assigned to the image
		self.subjects = []
		# classes
		self.classes = []

		for c,class_label in enumerate(self.class_labels,start=1):
			image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(class_label)]

			for image_path in image_paths:
#				print 'Image: ' + image_path
				# Read the image and convert to grayscale
				image_pil = Image.open(image_path).convert('L')
				# Convert the image format into numpy array
				image = np.array(image_pil, 'uint8')
				# Get the label of the image
				nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))

				self.images.append(image)
				self.subjects.append(nbr)
				self.classes.append(class_label)

			print 'class_label: {0}({1}#) - {2}'.format(c,len(image_paths), class_label)
