from rindex28_loader import Rindex28Loader
from fingerprint_recognizer import FingerprintRecognizer


rindex28 = Rindex28Loader('./databases/rindex28')
FingerprintRecognizer(rindex28.images, rindex28.subjects)
