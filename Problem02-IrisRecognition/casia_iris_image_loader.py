import os
from PIL import Image
import numpy as np


class ImageLoaderCASIAIris(object):
    """docstring for ImageLoaderCASIAIris"""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.get_images_and_subjects(self.path)

    def get_images_and_subjects(self, path):
        self.images = []
        self.subjects = []
        for subject_dir in self.get_directories_from(path):
            _, subject = os.path.split(subject_dir)
            subject = int(subject)
            eye_dirs = self.get_directories_from(subject_dir)
            for left_and_right_eye_dir in eye_dirs:
                for image_path in self.get_files_from(left_and_right_eye_dir):
                    termination = str.split(image_path, '.')[-1]
                    if termination == 'jpg':
                        l_or_r_eye = str.split(left_and_right_eye_dir, '/')[-1]
                        image_pil = Image.open(image_path).convert('L')
                        image_np = np.array(image_pil, 'uint8')
                        self.images.append(image_np)
                        self.subjects.append(str(subject)+l_or_r_eye)

    def get_directories_from(self, path):
        directories = []
        for element in os.listdir(path):
            path_to_element = os.path.join(path, element)
            if os.path.isdir(path_to_element):
                directories.append(path_to_element)
        return directories

    def get_files_from(self, path):
        files = []
        for element in os.listdir(path):
            path_to_element = os.path.join(path, element)
            if os.path.isfile(path_to_element):
                files.append(path_to_element)
        return files
