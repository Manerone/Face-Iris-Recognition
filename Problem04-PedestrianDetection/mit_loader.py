import os
from PIL import Image
import numpy as np
from sys import stdout


CLEAR_LINE = '\x1b[2K'

class MITLoader:
    """docstring for MITLoader"""
    def __init__(self, path):
        self.path = os.path.abspath(path)
        stdout.write('\rLoading database...')
        stdout.flush()
        self.get_images(self.path)
        stdout.write('\r' + CLEAR_LINE)
        stdout.flush()
    
    def get_images(self, path):
        self.images = []
        for image_path in self.get_files_from(path):
            image_pil = Image.open(image_path)
            image_np = np.array(image_pil, 'uint8')
            self.images.append(image_np)

    def get_files_from(self, path):
        files = []
        for element in os.listdir(path):
            path_to_element = os.path.join(path, element)
            if os.path.isfile(path_to_element):
                files.append(path_to_element)
        return np.sort(files)