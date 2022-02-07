# Does the same thing as main but groups all the channels together

# check Pillow version number
from PIL import Image, ImageOps, ImageFilter
import numpy as np

# K means, assign everything to nearest cluster (abs value of difference in this case)
# compute new averages

num_clusters = 2 # higher num of clusters here does better
num_iterations = 10
numChannels = 1
blur = True
grayscale_image = ImageOps.grayscale(Image.open("cartoon.jpeg"))
# Optional blur
if blur:
    grayscale_image = grayscale_image.filter(ImageFilter.GaussianBlur(20))
    grayscale_image.save('images/blurredCartoon.jpg')
im = np.asarray(grayscale_image)
color_im = np.zeros((len(im), len(im[0]), 3))
numPixels = im.flatten().size
picture_shape = np.shape(im[:, :])
shuffle = False

clusters = np.random.uniform(low=0.0, high=255.0, size=(numChannels, num_clusters))

channels = np.zeros((numChannels, numPixels), dtype=float)
channels[0] = im[:, :].flatten()

def getNearestCluster(channels):
    currentDifferences = np.zeros((num_clusters, numPixels))
    for i in range(num_clusters):
        for y in range(numChannels):
             currentDifferences[i][:] += pow((channels[y][:] - clusters[y][i]), 2)

    return currentDifferences.argmin(axis=0)


def getNewCluster(channels, clustersAtIndices):
    for i in range(num_clusters):
        indices = np.where(clustersAtIndices == i)
        for j in range(numChannels):
            values = channels[j][indices]
            if values.size == 0:
                clusters[j][i] = 0
            else:
                clusters[j][i] = np.average(values)

def process():
    for i in range(num_iterations):
        print('iteration: ', i)
        clustersAtIndices = getNearestCluster(channels)
        getNewCluster(channels, clustersAtIndices)


def createImageBW(finalClusters, indicesToCluster):
    for x in range(numChannels):
        final_channel_colors = finalClusters[x][indicesToCluster]
        im[:, :] = np.reshape(final_channel_colors, picture_shape)


def createImage(finalClusters, indicesToCluster):
    for x in range(3): #assumes 3
        final_channel_colors = finalClusters[x][indicesToCluster]
        color_im[:, :, x] = np.reshape(final_channel_colors, picture_shape)

# Used to convert the b/w image to different shades of one color (color channels scale 0 to 1)
def updateClustersWithColor(color):
    # Instead of multiplying everything by color, break it up by cluster
    # and scale the color uniformly
    sortedClusterIndices = np.argsort(clusters)[0]

    colorClusters = np.zeros((3, num_clusters))
    for i in range(num_clusters):
        shade = (i / float(num_clusters)) * color * 255
        ind = int(sortedClusterIndices[i])
        colorClusters[:, ind] = shade
    return colorClusters

def imageInOneColor(color):
    process()
    newClusters = updateClustersWithColor(color)
    createImage(newClusters, getNearestCluster(channels))
    x = Image.fromarray((color_im).astype(np.uint8))
    x.save('images/oneColor.jpg')

def bwImage():
    process()
    indicesToCluster = getNearestCluster(channels)
    createImageBW(clusters, indicesToCluster)
    x = Image.fromarray(im)
    x.save('bw2.jpg')

def outlineImage():

    process()
    indicesToCluster = getNearestCluster(channels)
    createImageBW(clusters, indicesToCluster)
    # im has the black and white image
    outlineImage = np.copy(im)
    for i in range(1, len(im) - 1):
        for j in range(1, len(im[0]) - 1):
            # check left right / up down for a difference
            if im[i, j - 1] != im[i, j + 1] or im[i - 1, j] != im[i + 1, j]:
                outlineImage[i, j] = 0
            # check diagonal
            elif im[i - 1, j - 1] != im[i + 1, j + 1] or im[i - 1, j + 1] != im[i + 1, j - 1]:
                outlineImage[i, j] = 0
            else:
                outlineImage[i, j] = 255
    x = Image.fromarray(outlineImage)
    x.save('images/cartoonOutline.jpg')

'''
unique_vals = set()
for i in im.flatten():
    if (i in unique_vals):
        continue
    else:
        print(i)
    unique_vals.add(i)
    '''



color = np.array([1.00, 0.0, 1.00])
#imageInOneColor(color)
outlineImage()
#bwImage()




