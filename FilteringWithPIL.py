from PIL import Image
from PIL import ImageFilter


class FilteringWithPIL:

    def __init__(self, imagePath):
        self.image = Image.open(imagePath)

    def getContoured(self):
        return self.image.filter(ImageFilter.CONTOUR)

    def getDetailed(self):
        return self.image.filter(ImageFilter.DETAIL)

    def getEdgeEnhanced(self):
        return self.image.filter(ImageFilter.EDGE_ENHANCE)

    def getEdgeEnhancedMore(self):
        return self.image.filter(ImageFilter.EDGE_ENHANCE_MORE)

    def getEmbossed(self):
        return self.image.filter(ImageFilter.EMBOSS)

    def getWithFoundEdges(self):
        return self.image.filter(ImageFilter.FIND_EDGES)

    def getSmoothed(self):
        return self.image.filter(ImageFilter.SMOOTH)

    def getSmoothedMore(self):
        return self.image.filter(ImageFilter.SMOOTH_MORE)

    def getSharpened(self):
        return self.image.filter(ImageFilter.SHARPEN)

    def getFilteredWithCustomKernel(self, kernel):
        size = (3, 3)
        kerFilter = ImageFilter.Kernel(size, kernel, scale=None, offset=0)
        return self.image.filter(kerFilter)
