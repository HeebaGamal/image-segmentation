import glob
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from matplotlib.image import imread
import matplotlib.pyplot as plt
images=glob.glob('images/*.png')
for i in images:
    image=imread(i)
    plt.figure(figsize=[12,6])
    plt.imshow(image)
    x=image.reshape([-1,3])
    bandwidth = estimate_bandwidth(x, quantile=.2,n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(x)
    image_seg=ms.cluster_centers_
    print(image_seg)
    print("================")
    image_seg=image_seg[ms.labels_]
    print(ms.labels_)
    image_seg=image_seg.reshape(image.shape)
    image_seg=image_seg
    print(image_seg.shape)
    #plt.figure(figsize=[12,6])
    plt.imshow(image_seg)
    plt.savefig('res/MeanShift/'+i)
    #plt.waitforbuttonpress(0)
