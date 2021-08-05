import glob
from sklearn.cluster import KMeans
from matplotlib.image import imread
import matplotlib.pyplot as plt
import datetime

images=glob.glob('images/*.png')
for i in images:
    image=imread(i)
    plt.figure(figsize=[12,6])
    plt.imshow(image)
    #(row,col,channel) => (row*col , channel)
    x=image.reshape([-1,3])
    km=KMeans(n_clusters=3)
    km.fit(x)
    image_seg=km.cluster_centers_
    print(image_seg)
    print("================")
    image_seg=image_seg[km.labels_]
    print(km.labels_)
    image_seg=image_seg.reshape(image.shape)
    print(image_seg.shape)
    plt.figure(figsize=[12,6])
    plt.imshow(image_seg)
    plt.savefig('res/Kmeans/'+i)

