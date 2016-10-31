from mit_loader import MITLoader
from pedestrian_detector import PedestrianDetector


loader = MITLoader('./databases/pedestrians128x64/')
detector = PedestrianDetector(loader.images)
detector.train()