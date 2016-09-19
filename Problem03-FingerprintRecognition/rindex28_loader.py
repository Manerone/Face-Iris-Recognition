import os
from PIL import Image
import numpy as np
from sys import stdout

CLEAR_LINE = '\x1b[2K'


class Rindex28Loader(object):
    """docstring for Rindex28Loader"""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        stdout.write('\rLoading database...')
        stdout.flush()
        self.get_images_and_subjects(self.path)
        stdout.write('\r' + CLEAR_LINE)
        stdout.flush()

    def get_images_and_subjects(self, path):
        self.images = []
        self.subjects = []
        for finger_image_path in self.get_files_from(path):
            subject = self.get_subject_from(finger_image_path)
            image_pil = Image.open(finger_image_path).convert('L')
            image_np = np.array(image_pil, 'uint8')
            self.subjects.append(subject)
            self.images.append(image_np)


    def get_subject_from(self, path):
        _, last_part = os.path.split(path)
        last_part = last_part[1:]
        last_part = str.split(last_part, '.')[0]
        return str.split(last_part, 'R')[0]

    def get_files_from(self, path):
        files = []
        for element in os.listdir(path):
            path_to_element = os.path.join(path, element)
            if os.path.isfile(path_to_element):
                files.append(path_to_element)
        return files