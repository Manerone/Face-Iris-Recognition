from faceid_database import YaleFaces
from faceid_database import ORL
from eigenface import Eigenface
import matplotlib.pyplot as plt

yale = YaleFaces('./databases/yalefaces/')
recognizer = Eigenface(yale.images)
images = recognizer.train()
for image in images:
	print image
