from skimage import io
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
# import time
import os

def init_image(filename):
    img = io.imread(filename)
    directory = filename.split('.')[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    io.imsave(directory + '/original.png', img)
    return os.stat(filename).st_size / 1024 ## KBytes


def cluster(filename, num_clusters, method):
    ''' This function runs the KMeans clustering algorithm '''

    # Load Image	
    img = io.imread(filename)
    img_size = img.shape

    # Reshape it to be 2-dimension
    X = img.reshape(img_size[0] * img_size[1], img_size[2])

    # Run the Kmeans algorithm
    km = KMeans(n_clusters=num_clusters, n_init=3, init=method, max_iter=300)
    km.fit(X)

    # Use the centroids to compress the image
    X_compressed = km.cluster_centers_[km.labels_]
    X_compressed = np.clip(X_compressed.astype('uint8'), 0, 255)

    # Reshape X_recovered to have the same dimension as the original image 128 * 128 * 3
    X_compressed = X_compressed.reshape(img_size[0], img_size[1], img_size[2])

    
    directory = filename.split('.')[0]
    save_path = directory + '/{}-{}-compressed.png'.format(method, num_clusters)
    io.imsave(save_path, X_compressed)
    return os.stat(save_path).st_size / 1024 ## KBytes




