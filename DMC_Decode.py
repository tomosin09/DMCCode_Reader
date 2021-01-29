import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
# Импортируем библиотеки
from sklearn.cluster import DBSCAN


# Select points on the image
def binarization(img):
    img_gray = cv.cvtColor(img,
                           cv.COLOR_BGR2GRAY)
    template = cv.imread('images/temp.jpg', 0)

    res = cv.matchTemplate(img_gray, template,
                           cv.TM_CCOEFF_NORMED)
    ret, thr = cv.threshold(res, 0.7, 255,
                            cv.THRESH_BINARY)
    thr = thr.astype('uint8')
    cnts = cv.findContours(thr, cv.RETR_LIST,
                           cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        # Obtain bounding rectangle to get measurements
        x, y, w, h = cv.boundingRect(c)
        # Find centroid
        M = cv.moments(c)
        # Check division by zero
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        pts.append([cX, cY])
        # Draw the contour and center of the shape on the image
        sv = cv.circle(thr, (cX, cY),
                       7, (255, 255, 255), -1)
    return res, thr


# Points clustering
def clust(pts):
    points = np.array(pts)
    db = DBSCAN(eps=32, min_samples=3).fit(points)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    unique_labels = set(labels)
    clusters = [[] for i in unique_labels]
    for i, j in zip(points, labels):
        if j != -1:
            clusters[j].append(i)
    return clusters


# Definition bounding box
def bbox(cl):
    print(len(cl))
    for i in cl:
        if len(i) == 0:
            continue
        else:
            rect = cv.minAreaRect(np.float32(i))
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(img3, [box], 0, (255, 0, 255), 2)




pts = []
# img_rgb = cv.imread('images/sample.jpg')
img_rgb = cv.imread('images/testlol.jpg')
img2 = img_rgb.copy()
res, thr = binarization(img_rgb)
cl = clust(pts)
colors = np.random.randint(30, 255, (len(cl), 3))
for i in range(len(cl)):
    i += 1
    for x in cl[i - 1]:
        cv.circle(img2, (x[0], x[1]), 7, (int(colors[i - 1][0]), int(colors[i - 1][1]), int(colors[i - 1][2])), -1)
img3 = img2.copy()
bbox(cl)
titles = ['Orig', 'Correlation', 'Thr', 'Clustering', 'BBox']
images = [img_rgb, res, thr, img2, img3]
# cv.imwrite('res.png', img2)
for i in range(5):
    plt.subplot(3, 2, i + 1);
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
