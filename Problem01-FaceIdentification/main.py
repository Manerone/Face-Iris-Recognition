from faceid_database import YaleFaces
from faceid_database import ORL
from eigenface import Eigenface
import matplotlib.pyplot as plt

yale = YaleFaces('./databases/yalefaces/')
eigenfaces = Eigenface.calculate(yale.images)
for eigenface in eigenfaces:
    plt.imshow(eigenface)
    plt.show()
