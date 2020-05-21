import scipy.ndimage
from PIL import Image
from pylab import *


class MedianFilter:

    def __init__(self, imagePath):
        self.image = array(Image.open(imagePath))

    def getMedianFiltered(self, size):
        im = scipy.ndimage.filters.median_filter(self.image, size=size, footprint= None, output=None, mode='reflect', cval=0.0, origin=0)
        return im