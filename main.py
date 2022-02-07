# check Pillow version number
from PIL import Image
import numpy as np

# K means, assign everything to nearest cluster (abs value of difference in this case)
# compute new averages

# This has more accurate results than others because it really has more colors possible

# TODO: can parallelize by channel

num_clusters = 20
num_iterations = 40
numChannels = 3

clusters = np.random.uniform(low=0.0, high=255.0, size=(numChannels, num_clusters))

# an array of numbers
def getNearestCluster(array, channel):
    currentDifferences = np.zeros((num_clusters, len(array)))
    for i in range(num_clusters):
        currentDifferences[i] = np.asarray(abs(array - float(clusters[channel][i])))

    return currentDifferences.argmin(axis=0)


def getNewCluster(colorArray, clustersAtIndices, channel):
    for i in range(num_clusters):
        indices = np.where(clustersAtIndices == i)
        values = colorArray[indices]
        if values.size == 0:
            clusters[channel][i] = 0
        else:
            clusters[channel][i] = np.average(values)


im = np.asarray(Image.open("greenApple.jpg"))

numPixels = im[:, :, 0].flatten().size


channels = np.zeros((3, numPixels), dtype=float)
channels[0] = im[:, :, 0].flatten()
channels[1] = im[:, :, 1].flatten()
channels[2] = im[:, :, 2].flatten()

for x in range(numChannels):
    for i in range(num_iterations):
        print('iteration: ', i)
        clustersAtIndices = getNearestCluster(channels[x], x)
        getNewCluster(channels[x], clustersAtIndices, x)

picture_shape = np.shape(im[:, :, 0])

for x in range(numChannels):
    final_channel_colors = clusters[x][getNearestCluster(channels[x], x)]
    im[:, :, x] = np.reshape(final_channel_colors, picture_shape)

print('Done')

unique_vals = set()
for x in im[:, :]:
    for y in x:
        z = (y[0], y[1], y[2])
        unique_vals.add(z)
print(len(unique_vals))

x = Image.fromarray(im)
x.save('im.jpg')




