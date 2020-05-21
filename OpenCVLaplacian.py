import cv2


class OpenCVLaplacian:

    def __init__(self, imagePath):
        self.image = cv2.imread(imagePath)

    def getLaplacian(self, ddepth):
        return cv2.Laplacian(self.image, ddepth)

