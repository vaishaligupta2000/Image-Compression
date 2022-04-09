from efficient import cluster, init_image
import matplotlib.pyplot  as plt
import sys
import time

## data variables
cluster_sizes = [1, 2, 4, 8, 16, 32, 64]
time_kmeans = []
time_kmeans_plus = []
image_sizes = []

## Initalise Image Folder
filename = sys.argv[1]
directory = filename.split('.')[0]
original_size = int(init_image(filename))

## Get Time for KMeans
for num_clusters in cluster_sizes:
	# Random
	start_time = time.time()
	cluster(filename, num_clusters, method = 'random')
	end_time = time.time()
	total_time = end_time - start_time
	time_kmeans.append(total_time)
	print('Kmeans', num_clusters, total_time)

	# KMeans++
	start_time = time.time()
	size = cluster(filename, num_clusters, method = 'k-means++')
	end_time = time.time()
	total_time = end_time - start_time
	time_kmeans_plus.append(total_time)
	print('Kmeans++', num_clusters, total_time)

	image_sizes.append(size)
	print('Size', num_clusters, size)


## Plot Time Taken
plt.figure(0)
plt.plot(cluster_sizes, time_kmeans, color = 'r' , label = 'K-means')
plt.plot(cluster_sizes, time_kmeans_plus,color='#037ffc', label = 'K-means++')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Time Taken')
plt.title('Time taken by K-means vs K-means++')
plt.legend()
plt.savefig(directory + '/time_analysis.png')


plt.figure(1)
plt.plot(cluster_sizes, image_sizes, color = 'r' , label = 'K-means')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Compressed Image Size - Original {} KB'.format(original_size))
plt.title('Size Comparison vs Number of KMeans clusters')
plt.legend()
plt.savefig(directory + '/size_analysis.png')
