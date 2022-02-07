# Does the same thing as main but groups all the channels together

# check Pillow version number
from PIL import Image
import numpy as np
import imageio

# K means, assign everything to nearest cluster (abs value of difference in this case)
# compute new averages

num_clusters = 2 # higher num of clusters here does better
num_iterations = 5
numChannels = 3
im = np.asarray(Image.open("elephants.jpg"))
numPixels = im[:, :, 0].flatten().size
picture_shape = np.shape(im[:, :, 0])
shuffle = False

clusters = np.random.uniform(low=0.0, high=255.0, size=(numChannels, num_clusters))

channels = np.zeros((3, numPixels), dtype=float)
channels[0] = im[:, :, 0].flatten()
channels[1] = im[:, :, 1].flatten()
channels[2] = im[:, :, 2].flatten()

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


def shuffleClusters():
    shuffled_indices = np.random.permutation(num_clusters)
    shuffled_clusters = np.zeros(clusters.shape)
    for i in range(num_clusters):
        ind = int(shuffled_indices[i])
        shuffled_clusters[:, i] = clusters[:, ind]
    return shuffled_clusters


def createImage(finalClusters, indicesToCluster):
    for x in range(numChannels):
        final_channel_colors = finalClusters[x][indicesToCluster]
        im[:, :, x] = np.reshape(final_channel_colors, picture_shape)

def countUniqueColors():
    unique_vals = set()
    for x in im[:, :]:
        for y in x:
            z = (y[0], y[1], y[2])
            unique_vals.add(z)
    print(len(unique_vals))

def main():
    process()
    indicesToCluster = getNearestCluster(channels)
    if shuffle:
        finalClusters = shuffleClusters()
    else:
        finalClusters = clusters
    createImage(finalClusters, indicesToCluster)
    print(countUniqueColors())
    x = Image.fromarray(im)
    x.save('something.jpg')

# TODO: this does it between iterations, try doing it running whole cluster and increasing each time
def generateGif():
    images = []

    for i in range(num_iterations):
        print('iteration: ', i)
        clustersAtIndices = getNearestCluster(channels)
        getNewCluster(channels, clustersAtIndices)
        # Intermittently do this to create gif
        createImage(clusters, clustersAtIndices)  # assigns current image to im
        images.append(np.copy(im))

    imageio.mimsave('gifs/example.gif3', images)

def generateGifByClusterSize():
    global num_clusters
    global clusters
    images = []

    for i in range(1, 5):
        print('NUM CLUSTERS: ', i)
        num_clusters = i
        clusters = np.random.uniform(low=0.0, high=255.0, size=(numChannels, num_clusters))
        process()
        indicesToCluster = getNearestCluster(channels)
        createImage(clusters, indicesToCluster)
        for x in range(10): # repeat the image so it lasts longer
            images.append(np.copy(im))
    for i in range(10, 100, 20):
        print('NUM CLUSTERS: ', i)
        num_clusters = i
        clusters = np.random.uniform(low=0.0, high=255.0, size=(numChannels, num_clusters))
        process()
        indicesToCluster = getNearestCluster(channels)
        createImage(clusters, indicesToCluster)
        padding = 10
        if i == 90:
            padding = 30
        for x in range(padding): # repeat the image so it lasts longer
            images.append(np.copy(im))
    imageio.mimsave('gifs/example5.gif', images)

# Used to convert the image to different shades of one color (color channels scale 0 to 1)
def overwriteClustersWithColor(color):
    for i in range(num_clusters):
        clusters[:, i] = i/float(num_clusters) * 255.0 * color


main()
#generateGifByClusterSize()



