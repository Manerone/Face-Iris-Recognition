from faceid_database import YaleFaces
from faceid_database import ORL
from eigenface import Eigenface
import matplotlib.pyplot as plt

yale = YaleFaces('./databases/yalefaces/')
eigenfaces = Eigenface.calculate(yale.images)
plt.imshow(eigenfaces[0], cmap='Greys_r')
plt.show()