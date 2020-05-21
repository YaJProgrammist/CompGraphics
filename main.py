from GrayLevelTransformation import GrayLevelTransformation
from MedianFilter import MedianFilter
from FilteringWithPIL import FilteringWithPIL
from OpenCVLaplacian import OpenCVLaplacian
from pylab import *


imagePath = 'images/fatcat.jpeg'


def testGrayLevelTransformation():

    imageForGrayLevelTransformation = GrayLevelTransformation(imagePath)

    figure(figsize=(15, 15))

    subplot(2, 2, 1)
    plt.imshow(imageForGrayLevelTransformation.image)
    plt.title('Original')

    subplot(2, 2, 2)
    plt.imshow(imageForGrayLevelTransformation.getNegated())
    plt.title('Negated')

    subplot(2, 2, 3)
    plt.imshow(imageForGrayLevelTransformation.getClampedToInterval(150, 250))
    plt.title('Clamped to interval 150..250')

    subplot(2, 2, 4)
    plt.imshow(imageForGrayLevelTransformation.getDarker(2))
    plt.title('Darker (degree = 2)')

    plt.show()


def testMedianFilter():

    imageForMedianFilter = MedianFilter(imagePath)

    figure(figsize=(15, 15))

    subplot(1, 2, 1)
    plt.imshow(imageForMedianFilter.image)
    plt.title('Original')

    subplot(1, 2, 2)
    plt.imshow(imageForMedianFilter.getMedianFiltered(5))
    plt.title('Median-filtered')

    plt.show()


def testFilteringWithPIL():

    imageForFilteringWithPIL = FilteringWithPIL(imagePath)

    figure(figsize=(15, 15))

    subplot(3, 4, 1)
    plt.imshow(imageForFilteringWithPIL.image)
    plt.title('Original')

    subplot(3, 4, 2)
    plt.imshow(imageForFilteringWithPIL.getContoured())
    plt.title('Contoured')

    subplot(3, 4, 3)
    plt.imshow(imageForFilteringWithPIL.getDetailed())
    plt.title('Detailed')

    subplot(3, 4, 4)
    plt.imshow(imageForFilteringWithPIL.getEdgeEnhanced())
    plt.title('Edge-enhanced')

    subplot(3, 4, 5)
    plt.imshow(imageForFilteringWithPIL.getEdgeEnhancedMore())
    plt.title('More edge-enhanced')

    subplot(3, 4, 6)
    plt.imshow(imageForFilteringWithPIL.getEmbossed())
    plt.title('Embossed')

    subplot(3, 4, 7)
    plt.imshow(imageForFilteringWithPIL.getWithFoundEdges())
    plt.title('With found edges')

    subplot(3, 4, 8)
    plt.imshow(imageForFilteringWithPIL.getSmoothed())
    plt.title('Smoothed')

    subplot(3, 4, 9)
    plt.imshow(imageForFilteringWithPIL.getSmoothedMore())
    plt.title('Smoothed more')

    subplot(3, 4, 10)
    plt.imshow(imageForFilteringWithPIL.getSharpened())
    plt.title('Sharpened')

    subplot(3, 4, 11)
    plt.imshow(imageForFilteringWithPIL.getFilteredWithCustomKernel([1, 1, 1, 1, -1, 1, -1, -1, -1]))
    plt.title('Filtered bwith custom kernel #1')

    subplot(3, 4, 12)
    plt.imshow(imageForFilteringWithPIL.getFilteredWithCustomKernel([1, 0, -1, 1, 0, -1, 0, 0, -1]))
    plt.title('Filtered with custom kernel #2')

    plt.show()


def testOpenCVLaplacian():

    imageForOpenCVLaplacian = OpenCVLaplacian(imagePath)

    figure(figsize=(15, 15))

    subplot(1, 2, 1)
    plt.imshow(imageForOpenCVLaplacian.image)
    plt.title('Original')

    subplot(1, 2, 2)
    plt.imshow(imageForOpenCVLaplacian.getLaplacian(-2))
    plt.title('Laplacian')

    plt.show()


testGrayLevelTransformation()
testMedianFilter()
testFilteringWithPIL()
testOpenCVLaplacian()