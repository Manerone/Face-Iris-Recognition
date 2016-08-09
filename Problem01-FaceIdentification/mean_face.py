from faceid_database import YaleFaces
from faceid_database import ORL
import numpy as np
import matplotlib.pyplot as plt

def calculate_mean_face(array_of_images):
    return (np.sum(array_of_images, 0))/len(array_of_images)

# Path to the Yale Dataset
path = './databases/yalefaces/'
print 'loading Yalefaces database'
yale = YaleFaces(path)

path = './databases/orl_faces/'
print 'loading ORL database'
orl = ORL(path)

mean_face_yale = calculate_mean_face(yale.images)
plt.imshow(mean_face_yale, cmap='Greys')
plt.show()
mean_face_orl = calculate_mean_face(orl.images)
plt.imshow(mean_face_orl, cmap='Greys')
plt.show()
