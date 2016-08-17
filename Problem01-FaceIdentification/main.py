from faceid_database import YaleFaces
from faceid_database import ORL
from eigenface import Eigenface
import matplotlib.pyplot as plt

yale = YaleFaces('./databases/yalefaces/')
recognizer = Eigenface(yale.images)
eigenfaces = recognizer.find_eigenfaces()
for eigenface in eigenfaces:
    plt.imshow(eigenface)
    plt.show()
