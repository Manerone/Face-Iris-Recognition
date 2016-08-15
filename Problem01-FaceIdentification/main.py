from faceid_database import YaleFaces
from faceid_database import ORL
from eigenface import Eigenface

yale = YaleFaces('./databases/yalefaces/')
Eigenface.calculate(yale.images)