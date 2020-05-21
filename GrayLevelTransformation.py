from PIL import Image
from pylab import *


class GrayLevelTransformation:

    def __init__(self, imagePath):
        self.image = array(Image.open(imagePath))

    def getNegated(self):
        im = 255 - self.image
        return im

    def getClampedToInterval(self, start, finish):
        clampRange = finish - start
        im = ((clampRange / 255) * self.image + start).astype(np.uint8)
        return im

    def getDarker(self, darknessDegree):
        im = (255.0 * (self.image / 255.0) ** darknessDegree).astype(np.uint8)
        return im
