import matplotlib.pyplot as plt
import numpy
from sklearn.cluster import KMeans

img = plt.imread("c.jpg")

height = img.shape[0]
width = img.shape[1]

img =img.reshape(height*width, 3)

kmeans = KMeans(n_clusters=15).fit(img)
labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_
img2 = numpy.zeros((height, width,3), dtype= numpy.uint8)

index = 0
for i in range (height):
	for j in range (width):
		img2[i][j]= clusters[labels[index]]
		index +=1

plt.imshow(img2)

plt.show()
plt.savefig('tessstttyyy.jpg')