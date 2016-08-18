from faceid_database import YaleFaces
from faceid_database import ORL
from eigenface import Eigenface
import matplotlib.pyplot as plt
import numpy as np

yale = YaleFaces('./databases/yalefaces/')
recognizer = Eigenface(yale.images)
recognizer.train(5)
reconstructed_img = recognizer.reconstruct_image(10)
plt.imshow(reconstructed_img)
plt.show()
